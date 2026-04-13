import jax
import jax.numpy as jnp
import optax
from optax import tree_utils as otu
from flax import nnx
from omegaconf import DictConfig
from typing import Optional, NamedTuple
import factorized, utils


def to_opt_state(tree):
    """Replacement for nnx.training.optimizer.to_opt_state that doesn't copy metadata."""

    def _to_opt_state(x):
        if isinstance(x, nnx.Variable):
            opt_state = nnx.OptVariable(x.value)
        else:
            opt_state = nnx.OptArray(x)
        return opt_state

    tree = jax.tree.map(_to_opt_state, tree, is_leaf=lambda x: isinstance(x, nnx.Variable))
    return tree


class ModelAndOptimizer(nnx.Optimizer):
    """
    Extends nnx.ModelAndOptimizer (v0.12.0) by:
    1) enabling stochastic rounding, and
    2) not copying model metadata onto optimizer (otherwise Adafactor fails with a sharded model).
    """

    def __init__(self, model, tx, wrt=nnx.Param, stochastic_round=False, rounding_flip_steps=0, lr_schedule=None):
        self.step = nnx.OptState(jnp.array(0, dtype=jnp.uint32))
        self.tx = tx
        raw_tx_state = tx.init(nnx.state(model, wrt))
        raw_tx_state = jax.tree.map(
            lambda x: x.astype(jnp.float32) if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating) else x,
            raw_tx_state,
        )
        self.opt_state = nnx.data(to_opt_state(raw_tx_state))
        self.wrt = wrt
        self.model = model
        self.stochastic_round = stochastic_round
        self.rounding_flip_steps = rounding_flip_steps
        self.lr_schedule = lr_schedule

    def update(self, key, grads, grad_dtype=None, report_update_metrics: bool = False, **kwargs):
        param_arrays = nnx.to_arrays(nnx.pure(nnx.state(self.model, self.wrt)))
        grad_arrays = nnx.to_arrays(nnx.pure(nnx.state(grads)))
        opt_state_arrays = nnx.to_arrays(nnx.pure(self.opt_state))
        kwargs_arrays = nnx.to_arrays(nnx.pure(kwargs))

        grad_arrays = jax.tree.map(lambda g: g.astype(grad_dtype), grad_arrays)

        updates, new_opt_state = self.tx.update(grad_arrays, opt_state_arrays, param_arrays, **kwargs_arrays)
        new_params = apply_updates(
            key,
            param_arrays,
            updates,
            self.stochastic_round,
            self.step.value,
            self.rounding_flip_steps,
        )

        update_cos_pre_post_cast = jnp.asarray(jnp.nan, dtype=jnp.float32)
        cast_error_mean = jnp.asarray(jnp.nan, dtype=jnp.float32)
        cast_bias_along_update = jnp.asarray(jnp.nan, dtype=jnp.float32)
        cast_error_rel_norm = jnp.asarray(jnp.nan, dtype=jnp.float32)
        update_cos_pre_post_cast_nolr = jnp.asarray(jnp.nan, dtype=jnp.float32)
        cast_bias_along_update_nolr = jnp.asarray(jnp.nan, dtype=jnp.float32)

        if report_update_metrics:
            metric_keys = otu.tree_split_key_like(key, param_arrays)

            def _materialize_low_precision(x):
                x_arr = jnp.asarray(x)
                if x_arr.dtype == jnp.bfloat16:
                    bits = jax.lax.bitcast_convert_type(x_arr, jnp.uint16)
                    return jax.lax.bitcast_convert_type(bits, jnp.bfloat16)
                return x_arr

            def _quantize_fp32_to_bf16_rne(x_f32):
                x_f32 = jnp.asarray(x_f32, dtype=jnp.float32)
                bits = jax.lax.bitcast_convert_type(x_f32, jnp.uint32)
                lsb = jax.lax.bitwise_and(
                    jax.lax.shift_right_logical(bits, jnp.uint32(16)),
                    jnp.uint32(1),
                )
                rounding_bias = jnp.uint32(0x7FFF) + lsb
                rounded = bits + rounding_bias
                bf16_bits = jax.lax.shift_right_logical(rounded, jnp.uint32(16)).astype(jnp.uint16)
                bf16_val = jax.lax.bitcast_convert_type(bf16_bits, jnp.bfloat16)
                return jax.lax.bitcast_convert_type(
                    jax.lax.bitcast_convert_type(bf16_val, jnp.uint16),
                    jnp.bfloat16,
                )

            def _quantize_endpoint(p_old_dtype, pre_cast_value, leaf_key):
                if self.stochastic_round and p_old_dtype == jnp.bfloat16:
                    return utils.to_bf16_stochastic(leaf_key, pre_cast_value)
                if self.rounding_flip_steps > 0 and p_old_dtype == jnp.bfloat16:
                    x_bits = jax.lax.bitcast_convert_type(pre_cast_value, jnp.uint32)
                    x_trunc = jax.lax.bitwise_and(x_bits, jnp.uint32(0xFFFF0000))
                    has_frac = jax.lax.bitwise_and(x_bits, jnp.uint32(0x0000FFFF)) != jnp.uint32(0)
                    x_away = x_trunc + jnp.where(has_frac, jnp.uint32(0x00010000), jnp.uint32(0))
                    use_away = ((self.step.value // self.rounding_flip_steps) % 2) == 1
                    result_bits = jnp.where(use_away, x_away, x_trunc)
                    return jax.lax.bitcast_convert_type(result_bits, jnp.float32).astype(jnp.bfloat16)
                if p_old_dtype == jnp.bfloat16:
                    return _quantize_fp32_to_bf16_rne(pre_cast_value)
                return pre_cast_value.astype(p_old_dtype)

            def _update_cos_leaf(p_old, u, leaf_key):
                if p_old is None:
                    return jnp.zeros((11,), dtype=jnp.float32)
                p_old = jnp.asarray(p_old)
                if not jnp.issubdtype(p_old.dtype, jnp.floating):
                    return jnp.zeros((11,), dtype=jnp.float32)

                pre_cast_update = jnp.asarray(u, dtype=jnp.float32)
                p_old_f32 = _materialize_low_precision(p_old).astype(jnp.float32)
                pre_cast = p_old_f32 + pre_cast_update
                p_new_metric = _quantize_endpoint(p_old.dtype, pre_cast, leaf_key)
                post_cast_update = _materialize_low_precision(p_new_metric).astype(jnp.float32) - p_old_f32
                cast_error = post_cast_update - pre_cast_update

                if self.lr_schedule is None:
                    pre_cast_update_nolr = pre_cast_update
                    post_cast_update_nolr = post_cast_update
                    cast_error_nolr = cast_error
                else:
                    lr_value = jnp.asarray(self.lr_schedule(self.step.value), dtype=jnp.float32)
                    safe_lr = jnp.where(jnp.abs(lr_value) > 0, lr_value, jnp.nan)
                    pre_cast_update_nolr = pre_cast_update / safe_lr
                    pre_cast_nolr = p_old_f32 + pre_cast_update_nolr
                    p_new_metric_nolr = _quantize_endpoint(p_old.dtype, pre_cast_nolr, leaf_key)
                    post_cast_update_nolr = _materialize_low_precision(p_new_metric_nolr).astype(jnp.float32) - p_old_f32
                    cast_error_nolr = post_cast_update_nolr - pre_cast_update_nolr

                return jnp.stack(
                    (
                        jnp.sum(pre_cast_update * post_cast_update),
                        jnp.sum(jnp.square(pre_cast_update)),
                        jnp.sum(jnp.square(post_cast_update)),
                        jnp.sum(cast_error),
                        jnp.asarray(pre_cast_update.size, dtype=jnp.float32),
                        jnp.sum(cast_error * pre_cast_update),
                        jnp.sum(jnp.square(cast_error)),
                        jnp.sum(pre_cast_update_nolr * post_cast_update_nolr),
                        jnp.sum(jnp.square(pre_cast_update_nolr)),
                        jnp.sum(jnp.square(post_cast_update_nolr)),
                        jnp.sum(cast_error_nolr * pre_cast_update_nolr),
                    )
                )

            update_cos_total = jax.tree.map(
                _update_cos_leaf,
                param_arrays,
                updates,
                metric_keys,
                is_leaf=lambda x: x is None,
            )
            update_cos_total = jax.tree.reduce(
                lambda a, b: a + b,
                update_cos_total,
                jnp.zeros((11,), dtype=jnp.float32),
            )
            update_cos_pre_post_cast = jnp.where(
                (update_cos_total[1] > 0) & (update_cos_total[2] > 0),
                update_cos_total[0] / (jnp.sqrt(update_cos_total[1]) * jnp.sqrt(update_cos_total[2])),
                jnp.nan,
            ).astype(jnp.float32)
            cast_error_mean = jnp.where(
                update_cos_total[4] > 0,
                update_cos_total[3] / update_cos_total[4],
                jnp.nan,
            ).astype(jnp.float32)
            cast_bias_along_update = jnp.where(
                update_cos_total[1] > 0,
                update_cos_total[5] / update_cos_total[1],
                jnp.nan,
            ).astype(jnp.float32)
            cast_error_rel_norm = jnp.where(
                update_cos_total[1] > 0,
                jnp.sqrt(update_cos_total[6] / update_cos_total[1]),
                jnp.nan,
            ).astype(jnp.float32)
            update_cos_pre_post_cast_nolr = jnp.where(
                (update_cos_total[8] > 0) & (update_cos_total[9] > 0),
                update_cos_total[7] / (jnp.sqrt(update_cos_total[8]) * jnp.sqrt(update_cos_total[9])),
                jnp.nan,
            ).astype(jnp.float32)
            cast_bias_along_update_nolr = jnp.where(
                update_cos_total[8] > 0,
                update_cos_total[10] / update_cos_total[8],
                jnp.nan,
            ).astype(jnp.float32)

            jax.debug.callback(
                lambda step, cos: print(f"[update_cos_pre_post_cast] step={int(step)} cos={float(cos):.6f}")
                if jax.process_index() == 0
                else None,
                self.step.value,
                update_cos_pre_post_cast,
            )
            jax.debug.callback(
                lambda step, mean_e, bias, rel: print(
                    f"[cast_error] step={int(step)} mean={float(mean_e):.3e} "
                    f"bias_along_update={float(bias):.3e} rel_norm={float(rel):.3e}"
                )
                if jax.process_index() == 0
                else None,
                self.step.value,
                cast_error_mean,
                cast_bias_along_update,
                cast_error_rel_norm,
            )
            jax.debug.callback(
                lambda step, cos: print(f"[update_cos_pre_post_cast_nolr] step={int(step)} cos={float(cos):.6f}")
                if jax.process_index() == 0
                else None,
                self.step.value,
                update_cos_pre_post_cast_nolr,
            )
            jax.debug.callback(
                lambda step, bias: print(
                    f"[cast_error_nolr] step={int(step)} bias_along_update={float(bias):.3e}"
                )
                if jax.process_index() == 0
                else None,
                self.step.value,
                cast_bias_along_update_nolr,
            )

        nnx.update(self.model, new_params)
        nnx.update(self.opt_state, nnx.state(new_opt_state))
        self.step[...] += 1
        return (
            update_cos_pre_post_cast,
            cast_error_mean,
            cast_bias_along_update,
            cast_error_rel_norm,
            update_cos_pre_post_cast_nolr,
            cast_bias_along_update_nolr,
        )


def apply_updates(
    key: jax.Array,
    params: optax.Params,
    updates: optax.Updates,
    stochastic_round=False,
    step=None,
    rounding_flip_steps=0,
) -> optax.Params:
    """Extends optax.apply_updates with stochastic rounding or flip rounding."""

    keys = otu.tree_split_key_like(key, params)
    _mask_upper = jnp.uint32(0xFFFF0000)
    _mask_lower = jnp.uint32(0x0000FFFF)
    _one_ulp = jnp.uint32(0x00010000)
    _zero_u32 = jnp.uint32(0)

    def leaf_update(p, u, key):
        if p is None:
            return None
        param_dtype = jnp.asarray(p).dtype
        p_f32 = p.astype(jnp.float32)
        u_f32 = jnp.asarray(u, dtype=jnp.float32)
        if stochastic_round and param_dtype == jnp.bfloat16:
            pre_cast = p_f32 + u_f32
            p_new = utils.to_bf16_stochastic(key, pre_cast)
        elif rounding_flip_steps > 0 and param_dtype == jnp.bfloat16:
            x = p_f32 + u_f32
            x_bits = jax.lax.bitcast_convert_type(x, jnp.uint32)
            x_trunc = jax.lax.bitwise_and(x_bits, _mask_upper)
            has_frac = jax.lax.bitwise_and(x_bits, _mask_lower) != _zero_u32
            x_away = x_trunc + jnp.where(has_frac, _one_ulp, _zero_u32)
            use_away = ((step // rounding_flip_steps) % 2) == 1
            result_bits = jnp.where(use_away, x_away, x_trunc)
            p_new = jax.lax.bitcast_convert_type(result_bits, jnp.float32).astype(jnp.bfloat16)
        else:
            p_new = (p_f32 + u_f32).astype(param_dtype)
        return p_new

    return jax.tree.map(leaf_update, params, updates, keys, is_leaf=lambda x: x is None)


def get_optimizer(c: DictConfig, num_opt_steps: int, tokens_per_opt_step: int, num_tokens_train: int | None = None):
    assert (c.peak_lr is not None) ^ ((c.peak_lr_scaled is not None) & (c.peak_lr_scaling is not None))
    if c.peak_lr is None:
        c.peak_lr = c.peak_lr_scaling * c.peak_lr_scaled

    warmup_steps = int(c.warmup_frac * num_opt_steps)
    end_lr = c.peak_lr * c.end_lr_frac
    base_lr_schedule = optax.schedules.warmup_cosine_decay_schedule(
        0, c.peak_lr, warmup_steps, num_opt_steps, end_value=end_lr
    )

    phase_batch_size = c.get("initial_batch_size")
    phase_tokens = int(c.get("initial_batch_tokens", 0) or 0)
    phase_peak_lr = c.get("initial_phase_peak_lr")
    phase_t1 = c.get("initial_phase_t1")
    phase_t2 = c.get("initial_phase_t2")
    switch_batch_size = c.get("switch_batch_size")
    switch_tokens = int(c.get("switch_at_tokens", 0) or 0)
    switch_peak_lr = c.get("switch_phase_peak_lr")
    switch_t1 = c.get("switch_phase_t1")
    switch_t2 = c.get("switch_phase_t2")
    seq_len = max(1, int(tokens_per_opt_step // int(c.batch_size)))
    if (
        phase_batch_size is not None
        and phase_peak_lr is not None
        and phase_tokens > 0
        and num_tokens_train is not None
    ):
        phase_batch_size = int(phase_batch_size)
        phase_tokens_per_step = phase_batch_size * seq_len
        phase_switch_steps = max(0, phase_tokens // phase_tokens_per_step)
        phase_total_steps = max(1, int(num_tokens_train) // phase_tokens_per_step)
        phase_warmup_steps = int(c.warmup_frac * phase_total_steps)
        phase_end_lr = float(phase_peak_lr) * c.end_lr_frac
        phase_lr_schedule = optax.schedules.warmup_cosine_decay_schedule(
            0,
            float(phase_peak_lr),
            phase_warmup_steps,
            phase_total_steps,
            end_value=phase_end_lr,
        )
        bs1_step_offset = max(0, phase_tokens // tokens_per_opt_step)

        def mixed_base_lr_schedule(step):
            step = jnp.asarray(step, dtype=jnp.int32)
            return jnp.where(
                step < phase_switch_steps,
                phase_lr_schedule(step),
                base_lr_schedule((step - phase_switch_steps) + bs1_step_offset),
            )
    elif (
        switch_batch_size is not None
        and switch_peak_lr is not None
        and switch_tokens > 0
        and num_tokens_train is not None
    ):
        switch_batch_size = int(switch_batch_size)
        switch_tokens_per_step = switch_batch_size * seq_len
        base_switch_steps = max(0, switch_tokens // tokens_per_opt_step)
        switch_total_steps = max(1, int(num_tokens_train) // switch_tokens_per_step)
        switch_warmup_steps = int(c.warmup_frac * switch_total_steps)
        switch_end_lr = float(switch_peak_lr) * c.end_lr_frac
        switch_lr_schedule = optax.schedules.warmup_cosine_decay_schedule(
            0,
            float(switch_peak_lr),
            switch_warmup_steps,
            switch_total_steps,
            end_value=switch_end_lr,
        )
        switch_step_offset = max(0, switch_tokens // switch_tokens_per_step)

        def mixed_base_lr_schedule(step):
            step = jnp.asarray(step, dtype=jnp.int32)
            return jnp.where(
                step < base_switch_steps,
                base_lr_schedule(step),
                switch_lr_schedule((step - base_switch_steps) + switch_step_offset),
            )
    else:
        mixed_base_lr_schedule = base_lr_schedule

    base_b1 = c.b1
    base_b2 = c.b2
    if base_b1 is None and c.t1 is not None:
        base_b1 = float(utils.halflife_to_decay(c.t1, tokens_per_opt_step))
    if base_b2 is None and c.t2 is not None:
        base_b2 = float(utils.halflife_to_decay(c.t2, tokens_per_opt_step))

    if (
        phase_batch_size is not None
        and phase_tokens > 0
        and num_tokens_train is not None
        and phase_t1 is not None
        and phase_t2 is not None
    ):
        phase_batch_size = int(phase_batch_size)
        phase_tokens_per_step = phase_batch_size * seq_len
        phase_switch_steps = max(0, phase_tokens // phase_tokens_per_step)
        phase_b1 = float(utils.halflife_to_decay(float(phase_t1), phase_tokens_per_step))
        phase_b2 = float(utils.halflife_to_decay(float(phase_t2), phase_tokens_per_step))

        def mixed_b1_schedule(step):
            step = jnp.asarray(step, dtype=jnp.int32)
            return jnp.where(step < phase_switch_steps, jnp.asarray(phase_b1, dtype=jnp.float32), jnp.asarray(base_b1, dtype=jnp.float32))

        def mixed_b2_schedule(step):
            step = jnp.asarray(step, dtype=jnp.int32)
            return jnp.where(step < phase_switch_steps, jnp.asarray(phase_b2, dtype=jnp.float32), jnp.asarray(base_b2, dtype=jnp.float32))
    elif (
        switch_batch_size is not None
        and switch_tokens > 0
        and num_tokens_train is not None
        and switch_t1 is not None
        and switch_t2 is not None
    ):
        switch_batch_size = int(switch_batch_size)
        switch_tokens_per_step = switch_batch_size * seq_len
        base_switch_steps = max(0, switch_tokens // tokens_per_opt_step)
        switch_b1 = float(utils.halflife_to_decay(float(switch_t1), switch_tokens_per_step))
        switch_b2 = float(utils.halflife_to_decay(float(switch_t2), switch_tokens_per_step))

        def mixed_b1_schedule(step):
            step = jnp.asarray(step, dtype=jnp.int32)
            return jnp.where(step < base_switch_steps, jnp.asarray(base_b1, dtype=jnp.float32), jnp.asarray(switch_b1, dtype=jnp.float32))

        def mixed_b2_schedule(step):
            step = jnp.asarray(step, dtype=jnp.int32)
            return jnp.where(step < base_switch_steps, jnp.asarray(base_b2, dtype=jnp.float32), jnp.asarray(switch_b2, dtype=jnp.float32))
    else:
        mixed_b1_schedule = None
        mixed_b2_schedule = None

    boost_factor = float(getattr(c, "init_lr_boost_factor", 1.0))
    boost_tokens = int(getattr(c, "init_lr_boost_tokens", 0) or 0)
    boost_steps = max(0, boost_tokens // tokens_per_opt_step)

    if boost_factor > 1.0 and boost_steps > 0:
        def lr_schedule(step):
            step = jnp.asarray(step, dtype=jnp.float32)
            base_lr = mixed_base_lr_schedule(step)
            progress = jnp.clip(step / jnp.float32(boost_steps), 0.0, 1.0)
            boost = 1.0 + (boost_factor - 1.0) * 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
            return base_lr * boost
    else:
        lr_schedule = mixed_base_lr_schedule

    assert (c.b1 is None) | (c.t1 is None)
    assert (c.b2 is None) | (c.t2 is None)
    assert (c.muon_b1 is None) | (c.muon_t1 is None)
    if c.b1 is None and c.t1 is not None:
        c.b1 = float(utils.halflife_to_decay(c.t1, tokens_per_opt_step))
    if c.b2 is None and c.t2 is not None:
        c.b2 = float(utils.halflife_to_decay(c.t2, tokens_per_opt_step))
    if c.t1 is None and c.b1 is not None:
        c.t1 = float(utils.decay_to_halflife(c.b1, tokens_per_opt_step))
    if c.t2 is None and c.b2 is not None:
        c.t2 = float(utils.decay_to_halflife(c.b2, tokens_per_opt_step))
    if c.muon_b1 is None and c.muon_t1 is not None:
        c.muon_b1 = float(utils.halflife_to_decay(c.muon_t1, tokens_per_opt_step))
    if c.muon_t1 is None and c.muon_b1 is not None:
        c.muon_t1 = float(utils.decay_to_halflife(c.muon_b1, tokens_per_opt_step))
    if c.b2_min is not None:
        c.b2 = max(c.b2, c.b2_min)

    if c.optimizer in ("sgd", "signum"):
        assert c.b2 is None
        assert c.t2 is None
        assert c.weight_decay == 0
        signed = c.optimizer == "signum"
        optimizer = sgd(lr_schedule, c.b1, signed)

    if c.optimizer == "adamw":
        assert c.b1 is not None
        assert c.b2 is not None
        if mixed_b1_schedule is not None and mixed_b2_schedule is not None:
            optimizer = optax.inject_hyperparams(optax.adamw)(
                learning_rate=lr_schedule,
                b1=mixed_b1_schedule,
                b2=mixed_b2_schedule,
                weight_decay=c.weight_decay,
            )
        else:
            optimizer = optax.adamw(lr_schedule, c.b1, c.b2, weight_decay=c.weight_decay)

    if c.optimizer == "adafactor":
        assert c.b1 is None
        assert c.b2 is not None
        assert c.weight_decay == 0
        optimizer = adafactor(lr_schedule, decay_rate=c.b2)

    if c.optimizer == "muon":
        assert c.b1 is not None
        assert c.b2 is not None
        assert c.muon_lr is not None
        assert c.muon_b1 is not None
        muon_lr = optax.schedules.warmup_cosine_decay_schedule(0, c.muon_lr, warmup_steps, num_opt_steps)
        optimizer = muon(muon_lr, c.muon_b1, lr_schedule, c.b1, c.b2)

    if c.clip_by_global_norm is not None:
        optimizer = optax.chain(optax.clip_by_global_norm(c.clip_by_global_norm), optimizer)

    return optimizer, lr_schedule


def sgd(
    learning_rate: optax.ScalarOrSchedule,
    b1: Optional[float] = None,
    signed=False,
) -> optax.GradientTransformation:
    return optax.chain(
        optax.trace(decay=b1) if b1 is not None else optax.identity(),
        optax.scale_by_sign() if signed else optax.identity(),
        optax.scale_by_learning_rate(learning_rate),
    )


def orthogonalize_via_newton_schulz(
    x: jax.Array,
    ns_coeffs: jax.Array,
    ns_steps: int = 5,
    eps: float = 1e-8,
) -> jax.Array:
    if x.ndim < 2:
        raise ValueError(f"Input must have >= 2 dims, got {x.shape}")
    if ns_coeffs.shape != (3,):
        raise ValueError(f"ns_coeffs must have shape (3,), got {ns_coeffs}")

    def newton_schulz_iterator(x: jax.Array, coeffs: jax.Array) -> jax.Array:
        x_mT = jnp.swapaxes(x, -2, -1)
        a = x @ x_mT
        b = coeffs[1] * a + coeffs[2] * a @ a
        return coeffs[0] * x + b @ x

    transposed = False
    if x.shape[-2] > x.shape[-1]:
        x = jnp.swapaxes(x, -2, -1)
        transposed = True
    x /= jnp.linalg.norm(x, axis=(-2, -1), keepdims=True) + eps
    ns_coeffs = ns_coeffs.astype(x.dtype)
    x = jax.lax.fori_loop(0, ns_steps, lambda _, x: newton_schulz_iterator(x, ns_coeffs), x)
    if transposed:
        x = jnp.swapaxes(x, -2, -1)
    return x


class MuonState(NamedTuple):
    """State for the Adam algorithm."""

    count: jax.Array
    mu: optax.Updates
    ns_coeffs: jax.Array


def scale_by_muon(
    ns_coeffs: tuple = (3.4445, -4.7750, 2.0315),
    ns_steps: int = 5,
    beta: float = 0.95,
    eps: float = 1e-8,
) -> optax.GradientTransformation:
    mu_dtype = None

    def init_fn(params):
        mu = otu.tree_zeros_like(params, dtype=mu_dtype)
        return MuonState(count=jnp.zeros([], jnp.int32), mu=mu, ns_coeffs=jnp.asarray(ns_coeffs))

    def update_fn(updates, state, params=None):
        del params
        mu = otu.tree_update_moment(updates, state.mu, beta, 1)
        updates = jax.tree.map(
            lambda u, m: orthogonalize_via_newton_schulz(m, state.ns_coeffs, ns_steps, eps)
            if u.ndim >= 2
            else u,
            updates,
            mu,
        )
        count_inc = optax.safe_int32_increment(state.count)
        return updates, MuonState(count_inc, mu, state.ns_coeffs)

    return optax.GradientTransformation(init_fn, update_fn)


def muon(
    learning_rate: float,
    beta: float,
    adam_learning_rate: float,
    adam_b1: float,
    adam_b2: float,
) -> optax.GradientTransformation:
    return optax.multi_transform(
        {
            "matrix": optax.chain(
                scale_by_muon(beta=beta),
                optax.scale_by_learning_rate(learning_rate),
            ),
            "vector": optax.chain(
                optax.scale_by_adam(adam_b1, adam_b2),
                optax.scale_by_learning_rate(adam_learning_rate),
            ),
        },
        param_labels=lambda x: jax.tree.map(lambda y: "matrix" if y.ndim >= 2 else "vector", x),
    )


def adafactor(
    learning_rate: optax.ScalarOrSchedule,
    decay_rate: float = 0.8,
    decay_offset: int = 0,
    min_dim_size_to_factor: int = 128,
    epsilon1: float = 1e-30,
    clipping_threshold: Optional[float] = None,
    momentum: Optional[float] = None,
    dtype_momentum: jnp.dtype = jnp.float32,
    weight_decay_rate: Optional[float] = None,
    eps: float = 1e-8,
    factored: bool = True,
) -> optax.GradientTransformation:
    return optax.chain(
        factorized.scale_by_factored_rms(
            factored=factored,
            decay_rate=decay_rate,
            decay_offset=decay_offset,
            min_dim_size_to_factor=min_dim_size_to_factor,
            epsilon1=epsilon1,
            clipping_threshold=clipping_threshold,
            momentum=momentum,
            dtype_momentum=dtype_momentum,
            weight_decay_rate=weight_decay_rate,
            eps=eps,
        ),
        optax.scale_by_learning_rate(learning_rate),
    )
