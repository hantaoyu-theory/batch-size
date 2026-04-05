import jax
import jax.numpy as jnp
import optax
from optax import tree_utils as otu
from flax import nnx
from jax.sharding import PartitionSpec as P
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
    def __init__(self, model, tx, wrt=nnx.Param, stochastic_round=False, rounding_flip_steps=0):
        self.step = nnx.OptState(jnp.array(0, dtype=jnp.uint32))
        self.tx = tx
        # Upcast float leaves to fp32 BEFORE wrapping in NNX containers, while
        # the optax state is still a plain pytree of jax arrays.
        raw_tx_state = tx.init(nnx.state(model, wrt))
        raw_tx_state = jax.tree.map(
            lambda x: x.astype(jnp.float32) if hasattr(x, 'dtype') and jnp.issubdtype(x.dtype, jnp.floating) else x,
            raw_tx_state,
        )
        self.opt_state = nnx.data(to_opt_state(raw_tx_state)) # <- CHANGED: doesn't copy metadata
        self.wrt = wrt
        self.model = model
        self.stochastic_round = stochastic_round
        self.rounding_flip_steps = rounding_flip_steps

    def update(self, key, grads, grad_dtype=None, report_zero_update_pct: bool = False, **kwargs):
        param_arrays = nnx.to_arrays(nnx.pure(nnx.state(self.model, self.wrt)))
        grad_arrays = nnx.to_arrays(nnx.pure(nnx.state(grads)))
        opt_state_arrays = nnx.to_arrays(nnx.pure(self.opt_state))
        kwargs_arrays = nnx.to_arrays(nnx.pure(kwargs))

        grad_arrays_raw = grad_arrays  # save pre-gcast grads for relative error metric
        grad_arrays = jax.tree.map(lambda g: g.astype(grad_dtype), grad_arrays)

        updates, new_opt_state = self.tx.update(grad_arrays, opt_state_arrays, param_arrays, **kwargs_arrays)
        new_params = apply_updates(key, param_arrays, updates, self.stochastic_round, self.step.value, self.rounding_flip_steps)

        # Metrics returned to the (jitted) caller so they can be logged to wandb in Python.
        # Tuple:
        # (zero_update_pct_nonzero_u, adam_snr,
        #  step_mean_abs_grad, step_mean_sq_grad, step_rms_grad,
        #  lost_abs_ratio, apply_efficiency,
        #  gcast_rel_error, pcast_rel_error,
        #  gcast_max_rel_error, pcast_max_rel_error,
        #  pcast_u_weighted_rel_error, pcast_frac_above_eps)
        zero_update_pct_nonzero_u = jnp.asarray(jnp.nan, dtype=jnp.float32)
        adam_snr = jnp.asarray(jnp.nan, dtype=jnp.float32)
        step_mean_abs_grad = jnp.asarray(jnp.nan, dtype=jnp.float32)
        step_mean_sq_grad = jnp.asarray(jnp.nan, dtype=jnp.float32)
        step_rms_grad = jnp.asarray(jnp.nan, dtype=jnp.float32)
        lost_abs_ratio = jnp.asarray(jnp.nan, dtype=jnp.float32)
        apply_efficiency = jnp.asarray(jnp.nan, dtype=jnp.float32)
        gcast_rel_error = jnp.asarray(jnp.nan, dtype=jnp.float32)
        pcast_rel_error = jnp.asarray(jnp.nan, dtype=jnp.float32)
        gcast_max_rel_error = jnp.asarray(jnp.nan, dtype=jnp.float32)
        pcast_max_rel_error = jnp.asarray(jnp.nan, dtype=jnp.float32)
        pcast_u_weighted_rel_error = jnp.asarray(jnp.nan, dtype=jnp.float32)
        pcast_frac_above_eps = jnp.asarray(jnp.nan, dtype=jnp.float32)

        # report the percentage of weights that are not updated after the update function
        if report_zero_update_pct:
            grad_leaves = [
                jnp.asarray(g, dtype=jnp.float32)
                for g in jax.tree.leaves(grad_arrays)
                if (g is not None) and hasattr(g, "dtype") and jnp.issubdtype(jnp.asarray(g).dtype, jnp.floating)
            ]
            if grad_leaves:
                total_grad = jnp.asarray(sum(int(g.size) for g in grad_leaves), dtype=jnp.float32)
                step_mean_abs_grad = sum(jnp.sum(jnp.abs(g)) for g in grad_leaves) / total_grad
                step_mean_sq_grad = sum(jnp.sum(jnp.square(g)) for g in grad_leaves) / total_grad
                step_rms_grad = jnp.sqrt(step_mean_sq_grad)

            def _count_leaf_unchanged(p_old, u, p_new):
                if p_old is None:
                    return jnp.zeros((2,), dtype=jnp.int32)
                p_old = jnp.asarray(p_old)
                if not jnp.issubdtype(p_old.dtype, jnp.floating):
                    return jnp.zeros((2,), dtype=jnp.int32)
                u = jnp.asarray(u)
                nonzero = (u != 0)
                unchanged = (p_new == p_old) & nonzero
                unchanged_count = jnp.sum(unchanged, dtype=jnp.int32)
                nonzero_count = jnp.sum(nonzero, dtype=jnp.int32)
                return jnp.stack((unchanged_count, nonzero_count))

            same_total = jax.tree.map(_count_leaf_unchanged, param_arrays, updates, new_params, is_leaf=lambda x: x is None)
            same_total = jax.tree.reduce(
                lambda a, b: a + b,
                same_total,
                jnp.zeros((2,), dtype=jnp.int32),
            )
            same, total = same_total[0], same_total[1]
            pct = jnp.where(
                total > 0,
                100.0 * (same.astype(jnp.float32) / total.astype(jnp.float32)),
                jnp.nan,
            )
            zero_update_pct_nonzero_u = pct.astype(jnp.float32)

            def _weighted_and_efficiency_stats_leaf(p_old, u, p_new):
                if p_old is None:
                    return jnp.zeros((4,), dtype=jnp.float32)
                p_old = jnp.asarray(p_old)
                if not jnp.issubdtype(p_old.dtype, jnp.floating):
                    return jnp.zeros((4,), dtype=jnp.float32)
                u = jnp.asarray(u, dtype=jnp.float32)
                p_new = jnp.asarray(p_new, dtype=jnp.float32)
                p_old_f32 = p_old.astype(jnp.float32)
                nonzero = (u != 0)
                unchanged = (p_new == p_old_f32) & nonzero
                abs_u = jnp.abs(u)
                # Cap |dp| at |u| so bf16 rounding amplification can't push apply_efficiency > 1.
                effective_dp = jnp.minimum(jnp.abs(p_new - p_old_f32), abs_u)
                # Returns: [sum|u|_lost, sum|u|_all, sum(min(|dp|,|u|)^2), sum(u^2)]
                return jnp.stack((
                    jnp.sum(jnp.where(unchanged, abs_u, 0.0)),
                    jnp.sum(abs_u),
                    jnp.sum(jnp.square(effective_dp)),
                    jnp.sum(jnp.square(u)),
                ))

            weighted_total = jax.tree.map(
                _weighted_and_efficiency_stats_leaf,
                param_arrays,
                updates,
                new_params,
                is_leaf=lambda x: x is None,
            )
            weighted_total = jax.tree.reduce(
                lambda a, b: a + b,
                weighted_total,
                jnp.zeros((4,), dtype=jnp.float32),
            )
            lost_abs_sum, abs_sum, dp_sq_sum, u_sq_sum = (
                weighted_total[0],
                weighted_total[1],
                weighted_total[2],
                weighted_total[3],
            )
            lost_abs_ratio = jnp.where(abs_sum > 0, lost_abs_sum / abs_sum, jnp.nan).astype(jnp.float32)
            apply_efficiency = jnp.where(
                u_sq_sum > 0,
                jnp.sqrt(dp_sq_sum / u_sq_sum),
                jnp.nan,
            ).astype(jnp.float32)
            jax.debug.callback(
                lambda step, pct: print(
                    f'[zero_update_pct_nonzero_u] step={int(step)} pct={float(pct):.2f}%'
                ) if jax.process_index() == 0 else None,
                self.step.value,
                pct,
            )
            jax.debug.callback(
                lambda step, lr, ae: print(
                    f'[update_quality] step={int(step)} lost_abs_ratio={float(lr):.3e} apply_efficiency={float(ae):.3e}'
                ) if jax.process_index() == 0 else None,
                self.step.value,
                lost_abs_ratio,
                apply_efficiency,
            )

            # Per-layer breakdown (grouped by parameter path).
            GetAttrKey = jax.tree_util.GetAttrKey
            DictKey = jax.tree_util.DictKey
            SequenceKey = jax.tree_util.SequenceKey

            def _key_name(k) -> str | None:
                if isinstance(k, GetAttrKey):
                    return k.name
                if isinstance(k, DictKey) and isinstance(k.key, str):
                    return k.key
                return None

            def _group_id_from_path(path) -> str:
                # Handle top-level modules.
                for k in path:
                    name = _key_name(k)
                    if name in ("token_embed_in", "token_embed_out", "out_ln"):
                        return name

                # Handle transformer blocks: blocks[i].*
                for i, k in enumerate(path):
                    is_blocks = (
                        (isinstance(k, GetAttrKey) and k.name == "blocks")
                        or (isinstance(k, DictKey) and k.key == "blocks")
                    )
                    if not is_blocks:
                        continue

                    if i + 1 >= len(path):
                        break
                    idx_key = path[i + 1]
                    if isinstance(idx_key, SequenceKey):
                        block = int(idx_key.idx)
                    if isinstance(idx_key, DictKey) and isinstance(idx_key.key, int):
                        block = int(idx_key.key)
                    if isinstance(idx_key, GetAttrKey):
                        # Some containers use attribute names like "_0".
                        if idx_key.name.startswith("_") and idx_key.name[1:].isdigit():
                            block = int(idx_key.name[1:])

                    if "block" in locals():
                        # Try to find the submodule within the block.
                        sub = "other"
                        # Common container/module names in this repo.
                        attn_names = {"attn", "qkv_proj", "out_proj", "query_norm", "key_norm", "attention"}
                        mlp_names = {"mlp", "fc1", "fc2"}
                        ln1_names = {"ln1"}
                        ln2_names = {"ln2"}
                        embed_names = {"token_embed_in", "token_embed_out"}

                        for kk in path[i + 2 :]:
                            name = _key_name(kk)
                            if name is None:
                                continue
                            if name in ln1_names:
                                sub = "ln1"
                                break
                            if name in ln2_names:
                                sub = "ln2"
                                break
                            if name in attn_names:
                                sub = "attn"
                                break
                            if name in mlp_names:
                                sub = "mlp"
                                break
                            if name in embed_names:
                                # Shouldn't generally happen inside blocks, but keep it explicit.
                                sub = name
                                break
                        return f"block_{block}.{sub}"

                return "other"

            # Flatten with paths so we can aggregate by layer id.
            param_path_leaves, _ = jax.tree_util.tree_flatten_with_path(param_arrays)
            update_path_leaves, _ = jax.tree_util.tree_flatten_with_path(updates)
            new_param_path_leaves, _ = jax.tree_util.tree_flatten_with_path(new_params)

            # Build a stable list of layer ids and sum counts into each.
            group_sums: dict[str, jax.Array] = {}
            for (p_path, p_leaf), (u_path, u_leaf), (n_path, n_leaf) in zip(
                param_path_leaves,
                update_path_leaves,
                new_param_path_leaves,
            ):
                # These should align structurally; if they don't, skip.
                if (p_path != u_path) or (p_path != n_path):
                    continue
                group = _group_id_from_path(p_path)
                counts = _count_leaf_unchanged(p_leaf, u_leaf, n_leaf)
                group_sums[group] = counts if group not in group_sums else (group_sums[group] + counts)

            # Print a compact one-line summary.
            if group_sums:
                sub_order = {"ln1": 0, "attn": 1, "ln2": 2, "mlp": 3, "other": 9}

                def _sort_key(name: str):
                    if name in ("token_embed_in", "token_embed_out"):
                        return (0, 0 if name == "token_embed_in" else 1, 0, 0)
                    if name == "out_ln":
                        return (1, 0, 0, 0)
                    if name.startswith("block_"):
                        # block_{i}.{sub}
                        block_part, _, sub = name.partition(".")
                        block_idx = block_part[6:]
                        if block_idx.isdigit():
                            return (2, int(block_idx), sub_order.get(sub, 99), 0)
                        return (2, 10_000_000, sub_order.get(sub, 99), 0)
                    if name == "other":
                        return (3, 0, 0, 0)
                    return (2, 10_000_001, 99, 0)

                group_names = sorted(group_sums.keys(), key=_sort_key)
                group_stats = jnp.stack(
                    [
                        jnp.stack(
                            (
                                jnp.where(
                                    group_sums[name][1] > 0,
                                    100.0
                                    * (
                                        group_sums[name][0].astype(jnp.float32)
                                        / group_sums[name][1].astype(jnp.float32)
                                    ),
                                    jnp.nan,
                                ),
                                group_sums[name][1].astype(jnp.float32),
                            )
                        )
                        for name in group_names
                    ],
                    axis=0,
                )

                def _print_group_stats(step, stats):
                    if jax.process_index() != 0:
                        return
                    # Print in chunks to keep lines readable.
                    parts = [f"{name}={float(stats[i,0]):.2f}%({int(stats[i,1])})" for i, name in enumerate(group_names)]
                    chunk = 8
                    for j in range(0, len(parts), chunk):
                        prefix = "[zero_update_pct_nonzero_u_by_group]" if j == 0 else " " * 33
                        print(f"{prefix} step={int(step)} " + ", ".join(parts[j:j+chunk]))

                jax.debug.callback(_print_group_stats, self.step.value, group_stats)

            # AdamW moments: mu (1st moment) and nu (2nd moment)
            state_path_leaves, _ = jax.tree_util.tree_flatten_with_path(new_opt_state)
            mu_leaves = []
            nu_leaves = []
            for path, leaf in state_path_leaves:
                if not hasattr(leaf, "dtype"):
                    continue
                leaf = jnp.asarray(leaf)
                if not jnp.issubdtype(leaf.dtype, jnp.floating):
                    continue
                key = jax.tree_util.keystr(path)
                if key.endswith(".mu") or (".mu" in key):
                    mu_leaves.append(leaf.astype(jnp.float32))
                if key.endswith(".nu") or (".nu" in key):
                    nu_leaves.append(leaf.astype(jnp.float32))

            if mu_leaves and nu_leaves and len(mu_leaves) == len(nu_leaves):
                total = jnp.asarray(sum(int(x.size) for x in mu_leaves), dtype=jnp.float32)
                eps_adam = 1e-8
                snr_sum = sum(
                    jnp.sum(jnp.abs(m) / (jnp.sqrt(n) + eps_adam))
                    for m, n in zip(mu_leaves, nu_leaves)
                )
                adam_snr = (snr_sum / total).astype(jnp.float32)

                jax.debug.callback(
                    lambda step, snr: print(
                        f'[adam_snr] step={int(step)} adam_snr={float(snr):.3e}'
                    ) if jax.process_index() == 0 else None,
                    self.step.value,
                    adam_snr,
                )

            if grad_leaves:
                jax.debug.callback(
                    lambda step, mag, msg, rmsg: print(
                        f'[step_grad_stats] step={int(step)} mean_abs_grad={float(mag):.3e} mean_sq_grad={float(msg):.3e} rms_grad={float(rmsg):.3e}'
                    ) if jax.process_index() == 0 else None,
                    self.step.value,
                    step_mean_abs_grad,
                    step_mean_sq_grad,
                    step_rms_grad,
                )

            # gcast relative error: mean(|G(g) - g| / (|g| + eps)) over all gradient elements
            _eps = jnp.asarray(1e-10, dtype=jnp.float32)
            raw_float_leaves = [
                jnp.asarray(g, dtype=jnp.float32)
                for g in jax.tree.leaves(grad_arrays_raw)
                if (g is not None) and hasattr(g, "dtype") and jnp.issubdtype(jnp.asarray(g).dtype, jnp.floating)
            ]
            cast_float_leaves = [
                jnp.asarray(g, dtype=jnp.float32)
                for g in jax.tree.leaves(grad_arrays)
                if (g is not None) and hasattr(g, "dtype") and jnp.issubdtype(jnp.asarray(g).dtype, jnp.floating)
            ]
            if raw_float_leaves and len(raw_float_leaves) == len(cast_float_leaves):
                total_g = jnp.asarray(sum(int(g.size) for g in raw_float_leaves), dtype=jnp.float32)
                rel_err_leaves = [
                    jnp.abs(gc - gr) / (jnp.abs(gr) + _eps)
                    for gr, gc in zip(raw_float_leaves, cast_float_leaves)
                ]
                gcast_rel_error = (
                    sum(jnp.sum(e) for e in rel_err_leaves) / total_g
                ).astype(jnp.float32)
                gcast_max_rel_error = jnp.max(
                    jnp.stack([jnp.max(e) for e in rel_err_leaves])
                ).astype(jnp.float32)
                jax.debug.callback(
                    lambda step, e, mx: print(
                        f'[gcast_rel_error] step={int(step)} mean={float(e):.3e} max={float(mx):.3e}'
                    ) if jax.process_index() == 0 else None,
                    self.step.value,
                    gcast_rel_error,
                    gcast_max_rel_error,
                )

            # pcast stats: mean, max, |u|-weighted mean, frac above bf16-eps threshold
            # Computed from deterministic pcast (does not use new_params to avoid CSE aliasing).
            # For bf16 params, use explicit bit truncation to avoid XLA folding
            # fp32->bf16->fp32 casts into identity on TPU.
            # Accumulator layout: [sum(rel_err), max(rel_err), count,
            #                      sum(rel_err*|u|), sum(|u|), sum(rel_err > 2e-3)]
            _bf16_eps_thresh = jnp.asarray(2e-3, dtype=jnp.float32)
            _bf16_mask = jnp.asarray(0xFFFF0000, dtype=jnp.uint32)

            def _pcast_rel_error_leaf(p_old, u):
                if p_old is None:
                    return jnp.zeros((6,), dtype=jnp.float32)
                p_old = jnp.asarray(p_old)
                if not jnp.issubdtype(p_old.dtype, jnp.floating):
                    return jnp.zeros((6,), dtype=jnp.float32)
                u_f32 = jnp.asarray(u, dtype=jnp.float32)
                x = p_old.astype(jnp.float32) + u_f32
                if p_old.dtype == jnp.bfloat16:
                    x_bits = jax.lax.bitcast_convert_type(x, jnp.uint32)
                    x_bits_trunc = jax.lax.bitwise_and(x_bits, _bf16_mask)
                    x_pcast = jax.lax.bitcast_convert_type(x_bits_trunc, jnp.float32)
                else:
                    x_pcast = x.astype(p_old.dtype).astype(jnp.float32)
                abs_u = jnp.abs(u_f32)
                rel_err = jnp.abs(x_pcast - x) / (jnp.abs(x) + _eps)
                return jnp.stack((
                    jnp.sum(rel_err),
                    jnp.max(rel_err),
                    jnp.asarray(p_old.size, dtype=jnp.float32),
                    jnp.sum(rel_err * abs_u),
                    jnp.sum(abs_u),
                    jnp.sum(jnp.where(rel_err > _bf16_eps_thresh, 1.0, 0.0)),
                ))

            pcast_totals = jax.tree.map(
                _pcast_rel_error_leaf, param_arrays, updates, is_leaf=lambda x: x is None
            )
            pcast_totals = jax.tree.reduce(
                lambda a, b: jnp.stack((
                    a[0] + b[0], jnp.maximum(a[1], b[1]), a[2] + b[2],
                    a[3] + b[3], a[4] + b[4], a[5] + b[5],
                )),
                pcast_totals,
                jnp.zeros((6,), dtype=jnp.float32),
            )
            pcast_rel_error = jnp.where(
                pcast_totals[2] > 0, pcast_totals[0] / pcast_totals[2], jnp.nan
            ).astype(jnp.float32)
            pcast_max_rel_error = pcast_totals[1].astype(jnp.float32)
            pcast_u_weighted_rel_error = jnp.where(
                pcast_totals[4] > 0, pcast_totals[3] / pcast_totals[4], jnp.nan
            ).astype(jnp.float32)
            pcast_frac_above_eps = jnp.where(
                pcast_totals[2] > 0, pcast_totals[5] / pcast_totals[2], jnp.nan
            ).astype(jnp.float32)
            jax.debug.callback(
                lambda step, e, mx, uw, fa: print(
                    f'[pcast_rel_error] step={int(step)} mean={float(e):.3e} max={float(mx):.3e}'
                    f' u_weighted={float(uw):.3e} frac_above_eps={float(fa):.3e}'
                ) if jax.process_index() == 0 else None,
                self.step.value,
                pcast_rel_error,
                pcast_max_rel_error,
                pcast_u_weighted_rel_error,
                pcast_frac_above_eps,
            )

        nnx.update(self.model, new_params)
        nnx.update(self.opt_state, nnx.state(new_opt_state))
        self.step[...] += 1
        return (
            zero_update_pct_nonzero_u,   # [0]
            adam_snr,                    # [1]
            step_mean_abs_grad,          # [2]
            step_mean_sq_grad,           # [3]
            step_rms_grad,               # [4]
            lost_abs_ratio,              # [5]
            apply_efficiency,            # [6]
            gcast_rel_error,             # [7]
            pcast_rel_error,             # [8]
            gcast_max_rel_error,         # [9]
            pcast_max_rel_error,         # [10]
            pcast_u_weighted_rel_error,  # [11]
            pcast_frac_above_eps,        # [12]
        )


def apply_updates(
    key: jax.Array,
    params: optax.Params,
    updates: optax.Updates,
    stochastic_round = False,
    step = None,
    rounding_flip_steps = 0,
) -> optax.Params:
    """Extends optax.apply_updates with stochastic rounding or flip rounding."""

    keys = otu.tree_split_key_like(key, params)
    _mask_upper = jnp.uint32(0xFFFF0000)
    _mask_lower = jnp.uint32(0x0000FFFF)
    _one_ulp = jnp.uint32(0x00010000)
    _zero_u32 = jnp.uint32(0)

    def leaf_update(p, u, key):
        if p is None: return None
        param_dtype = jnp.asarray(p).dtype
        if stochastic_round:
            p = p.astype(jnp.float32) + u
            p = utils.to_bf16_stochastic(key, p)
        elif rounding_flip_steps > 0 and param_dtype == jnp.bfloat16:
            # Alternate between truncation (toward zero) and round-away (away from zero)
            # every rounding_flip_steps steps. Over two periods the bias cancels.
            x = p.astype(jnp.float32) + u
            x_bits = jax.lax.bitcast_convert_type(x, jnp.uint32)
            x_trunc = jax.lax.bitwise_and(x_bits, _mask_upper)
            has_frac = (jax.lax.bitwise_and(x_bits, _mask_lower) != _zero_u32)
            x_away = x_trunc + jnp.where(has_frac, _one_ulp, _zero_u32)
            use_away = ((step // rounding_flip_steps) % 2) == 1
            result_bits = jnp.where(use_away, x_away, x_trunc)
            p = jax.lax.bitcast_convert_type(result_bits, jnp.float32).astype(jnp.bfloat16)
        else:
            p = (p.astype(jnp.float32) + u).astype(param_dtype)
        return p
    return jax.tree.map(leaf_update, params, updates, keys, is_leaf=lambda x: x is None)


def get_optimizer(c: DictConfig, num_opt_steps: int, tokens_per_opt_step: int):
    
    # get LR
    assert (c.peak_lr is not None) ^ ((c.peak_lr_scaled is not None) & (c.peak_lr_scaling is not None))
    if c.peak_lr is None:
        c.peak_lr = c.peak_lr_scaling * c.peak_lr_scaled

    # get schedule
    warmup_steps = int(c.warmup_frac * num_opt_steps)
    end_lr = c.peak_lr * c.end_lr_frac
    lr_schedule = optax.schedules.warmup_cosine_decay_schedule(0, c.peak_lr, warmup_steps, num_opt_steps, end_value=end_lr)

    # convert (t1 <-> b1), (t2 <-> b2)
    assert (c.b1 is None) | (c.t1 is None) # at most one can be specified in config
    assert (c.b2 is None) | (c.t2 is None) # at most one can be specified in config
    assert (c.muon_b1 is None) | (c.muon_t1 is None) # at most one can be specified in config
    if c.b1 is None and c.t1 is not None: c.b1 = float(utils.halflife_to_decay(c.t1, tokens_per_opt_step))
    if c.b2 is None and c.t2 is not None: c.b2 = float(utils.halflife_to_decay(c.t2, tokens_per_opt_step))
    if c.t1 is None and c.b1 is not None: c.t1 = float(utils.decay_to_halflife(c.b1, tokens_per_opt_step))
    if c.t2 is None and c.b2 is not None: c.t2 = float(utils.decay_to_halflife(c.b2, tokens_per_opt_step))
    if c.muon_b1 is None and c.muon_t1 is not None: c.muon_b1 = float(utils.halflife_to_decay(c.muon_t1, tokens_per_opt_step))
    if c.muon_t1 is None and c.muon_b1 is not None: c.muon_t1 = float(utils.decay_to_halflife(c.muon_b1, tokens_per_opt_step))
    if c.b2_min is not None: c.b2 = max(c.b2, c.b2_min)

    if c.optimizer in ('sgd', 'signum'):
        assert c.b2 is None
        assert c.t2 is None
        assert c.weight_decay == 0
        signed = c.optimizer == 'signum'
        optimizer = sgd(lr_schedule, c.b1, signed)

    if c.optimizer == 'adamw':
        assert c.b1 is not None
        assert c.b2 is not None
        optimizer = optax.adamw(lr_schedule, c.b1, c.b2, weight_decay=c.weight_decay)
    
    if c.optimizer == 'adafactor':
        assert c.b1 is None
        assert c.b2 is not None
        assert c.weight_decay == 0
        optimizer = adafactor(lr_schedule, decay_rate=c.b2)

    if c.optimizer == 'muon':
        assert c.b1 is not None
        assert c.b2 is not None
        assert c.muon_lr is not None
        assert c.muon_b1 is not None
        muon_lr = optax.schedules.warmup_cosine_decay_schedule(0, c.muon_lr, warmup_steps, num_opt_steps)
        optimizer = muon(muon_lr, c.muon_b1, lr_schedule, c.b1, c.b2)

    if c.clip_by_global_norm is not None:
        optimizer = optax.chain(optax.clip_by_global_norm(c.clip_by_global_norm), optimizer)

    return optimizer


def sgd(
    learning_rate: optax.ScalarOrSchedule,
    b1: Optional[float] = None,
    signed = False,
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
    # https://github.com/google-deepmind/optax/blob/main/optax/contrib/_muon.py 
    if x.ndim < 2:
        raise ValueError(f'Input must have >= 2 dims, got {x.shape}')
    if ns_coeffs.shape != (3,):
        raise ValueError(f'ns_coeffs must have shape (3,), got {ns_coeffs}')
    def newton_schulz_iterator(x: jax.Array, coeffs: jax.Array) -> jax.Array:
        x_mT = jnp.swapaxes(x, -2, -1) # <-- changed (matrix transpose last 2 dims)
        a = x @ x_mT # <-- changed (use matrix transpose)
        b = coeffs[1] * a + coeffs[2] * a @ a
        return coeffs[0] * x + b @ x
    transposed = False
    if x.shape[-2] > x.shape[-1]: # <-- changed (check last 2 dims)
        x = jnp.swapaxes(x, -2, -1) # <-- changed (transpose last 2 dims)
        transposed = True
    x /= (jnp.linalg.norm(x, axis=(-2, -1), keepdims=True) + eps) # <-- changed (normalize each matrix slice)
    ns_coeffs = ns_coeffs.astype(x.dtype)
    x = jax.lax.fori_loop(0, ns_steps, lambda _, x: newton_schulz_iterator(x, ns_coeffs), x)
    if transposed: x = jnp.swapaxes(x, -2, -1) # <-- changed (transpose last 2 dims)
    return x


class MuonState(NamedTuple):
    """State for the Adam algorithm."""
    count: jax.Array # shape=(), dtype=jnp.int32.
    mu: optax.Updates
    ns_coeffs: jax.Array # shape=(), dtype=jnp.int32.


def scale_by_muon(
    ns_coeffs: tuple = (3.4445, -4.7750, 2.0315),
    ns_steps: int = 5,
    beta: float = 0.95,
    eps: float = 1e-8,
) -> optax.GradientTransformation:
    # https://github.com/google-deepmind/optax/blob/main/optax/contrib/_muon.py 

    def init_fn(params):
        mu = otu.tree_zeros_like(params) # First moment
        return MuonState(jnp.zeros([], jnp.int32), mu, jnp.asarray(ns_coeffs))

    def update_fn(updates, state, params=None):
        del params
        mu = otu.tree_update_moment(updates, state.mu, beta, 1)
        count_inc = optax.safe_increment(state.count)
        mu_hat = otu.tree_bias_correction(mu, beta, count_inc)
        # Apply Newton-schulz orthogonalization.
        updates = jax.tree.map(lambda x: orthogonalize_via_newton_schulz(x, state.ns_coeffs, ns_steps, eps), mu_hat)
        updates = jax.tree.map(lambda x: jnp.sqrt(jnp.maximum(1, x.shape[-1] / x.shape[-2])) * x, updates)
        return updates, MuonState(count_inc, mu, state.ns_coeffs)
    
    return optax.GradientTransformation(init_fn, update_fn)


def muon(
    learning_rate: float,
    muon_b1: float,
    adam_lr: float,
    adam_b1: float,
    adam_b2: float,
) -> optax.GradientTransformation:
    return optax.multi_transform(
        transforms={
            'muon': optax.chain(
                scale_by_muon(beta=muon_b1),
                optax.scale_by_learning_rate(learning_rate),
            ),
            'adam': optax.adamw(adam_lr, adam_b1, adam_b2)
        },
        param_labels=lambda params: jax.tree.map_with_path(
            lambda path, val: 'adam' if 'embed' in jax.tree_util.keystr(path) else 'muon', params
        ),
    )


def adafactor(
    learning_rate: optax.ScalarOrSchedule,
    decay_rate: float = 0.8,
    clipping_threshold: Optional[float] = 1.0,
    min_dim_size_to_factor: int = 128,
) -> optax.GradientTransformation:
    """
    Adafactor reimplemented to use float32 state, regardless of param dtype.
    https://github.com/google-deepmind/optax/blob/8973bb3c77b07850737246815f1c028b53fffbe0/optax/_src/alias.py#L225#L327
    """
    return optax.chain(
        factorized.scale_by_factored_rms(decay_rate=decay_rate, min_dim_size_to_factor=min_dim_size_to_factor),
        optax.clip_by_block_rms(clipping_threshold) if clipping_threshold is not None else optax.identity(),
        optax.scale_by_learning_rate(learning_rate),
        optax.scale_by_param_block_rms(),
    )
