import math
import jax
import jax.numpy as jnp
import optax
import wandb
from functools import partial
from flax import nnx
from optax import tree_utils as otu
from tqdm.auto import tqdm
from omegaconf.dictconfig import DictConfig
import data, utils
import model as model_lib
import optimizer as optimizer_lib


@partial(jax.jit, static_argnames=('model_graphdef', 'pad'))
def loss_fn(model_state, model_graphdef, x, pad=False): # [B, T]
    model = nnx.merge(model_graphdef, model_state)
    y = jnp.roll(x, -1, axis=1)
    loss_mask = data.pad_mask(x) if pad else jnp.ones(x.shape, dtype=bool)
    loss_mask = loss_mask.at[:, -1].set(False)
    logits = model(x) # [B, T, V]
    losses = optax.softmax_cross_entropy_with_integer_labels(logits.astype(jnp.float32), y) # [B, T]
    return (losses * loss_mask).sum() / loss_mask.sum()


@partial(jax.jit, static_argnames=('opt_graphdef', 'model_graphdef', 'grad_dtype', 'report_zero_update_pct'))
def train_step(key, opt_state, opt_graphdef, model_graphdef, batch, grad_dtype=None, report_zero_update_pct: bool = False):
    key, key_opt = jax.random.split(key)

    def _cast_params_for_grad(model_state, target_grad_dtype):
        if target_grad_dtype is None:
            return model_state

        def _maybe_upcast_param(p):
            p_arr = jnp.asarray(p)
            if not jnp.issubdtype(p_arr.dtype, jnp.floating):
                return p
            if jnp.finfo(p_arr.dtype).bits < jnp.finfo(target_grad_dtype).bits:
                return p_arr.astype(target_grad_dtype)
            return p

        return jax.tree.map(_maybe_upcast_param, model_state)

    # compute grads from a single micro-batch
    if batch.ndim == 2:
        params_for_grad = _cast_params_for_grad(opt_state.model, grad_dtype)
        loss, grads = jax.value_and_grad(loss_fn)(params_for_grad, model_graphdef, batch)

    # compute grads from multiple micro-batches (using gradient accumulation)
    if batch.ndim == 3:
        loss = 0
        grads = otu.tree_zeros_like(opt_state.model, dtype=jnp.float32)
        params_for_grad = _cast_params_for_grad(opt_state.model, grad_dtype)

        def step_fn(i, args):
            loss, grads = args
            batch_loss, batch_grads = jax.value_and_grad(loss_fn)(params_for_grad, model_graphdef, batch[i])
            loss = (i*loss + batch_loss) / (i+1)
            grads = jax.tree.map(lambda m, g: (i*m + g) / (i+1), grads, batch_grads)
            return loss, grads
        loss, grads = jax.lax.fori_loop(0, batch.shape[0], step_fn, (loss, grads))

    optimizer = nnx.merge(opt_graphdef, opt_state)
    update_metrics = optimizer.update(
        key_opt,
        grads,
        grad_dtype=grad_dtype,
        report_zero_update_pct=report_zero_update_pct,
    )
    opt_state = nnx.state(optimizer)
    return key, opt_state, loss, update_metrics


@partial(jax.jit, static_argnames=('model_graphdef',))
def _compute_grads_tree(model_state, model_graphdef, batch):
    """Compute fp32 gradients for a model, returning the gradient pytree."""
    def _to_fp32(state):
        def _cast(x):
            x_arr = jnp.asarray(x)
            return x_arr.astype(jnp.float32) if jnp.issubdtype(x_arr.dtype, jnp.floating) else x
        return jax.tree.map(_cast, state)
    _, grads = jax.value_and_grad(loss_fn)(_to_fp32(model_state), model_graphdef, batch)
    return grads


def compute_grad_diff_metrics(grads_p, grads_s, debug=False):
    """Compare two gradient trees by matching leaves by path string."""
    def _to_dict(grads):
        d = {}
        for path, leaf in jax.tree_util.tree_flatten_with_path(grads)[0]:
            if leaf is not None and hasattr(leaf, 'dtype') and jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.floating):
                d[jax.tree_util.keystr(path)] = jnp.asarray(leaf, dtype=jnp.float32).ravel()
        return d

    p_dict = _to_dict(grads_p)
    s_dict = _to_dict(grads_s)

    if debug:
        p_keys = sorted(p_dict.keys())
        s_keys = sorted(s_dict.keys())
        print(f'[debug] primary has {len(p_keys)} grad leaves, shadow has {len(s_keys)} grad leaves')
        print(f'[debug] primary first 5 paths: {p_keys[:5]}')
        print(f'[debug] shadow  first 5 paths: {s_keys[:5]}')
        common = set(p_keys) & set(s_keys)
        print(f'[debug] common paths: {len(common)}, primary-only: {len(set(p_keys) - common)}, shadow-only: {len(set(s_keys) - common)}')

    common_keys = sorted(set(p_dict.keys()) & set(s_dict.keys()))
    if not common_keys:
        return float('nan'), float('nan'), float('nan')

    p_flat = jnp.concatenate([p_dict[k] for k in common_keys])
    s_flat = jnp.concatenate([s_dict[k] for k in common_keys])

    eps = 1e-12
    dot = float(jnp.sum(p_flat * s_flat))
    p_norm = float(jnp.sqrt(jnp.sum(jnp.square(p_flat))))
    s_norm = float(jnp.sqrt(jnp.sum(jnp.square(s_flat))))
    diff_norm = float(jnp.sqrt(jnp.sum(jnp.square(p_flat - s_flat))))

    cos = dot / (p_norm * s_norm + eps)
    rel_l2 = diff_norm / (s_norm + eps)

    nonzero = (p_flat != 0) | (s_flat != 0)
    sign_diff = nonzero & (jnp.sign(p_flat) != jnp.sign(s_flat))
    n_eligible = float(jnp.sum(nonzero))
    sign_pct = 100.0 * float(jnp.sum(sign_diff)) / n_eligible if n_eligible > 0 else float('nan')

    return cos, rel_l2, sign_pct


def eval_step(model_state, model_graphdef, dataset, pad=False):
    loss = 0
    for batch in dataset:
        loss += loss_fn(model_state, model_graphdef, batch, pad)
    return loss / len(dataset)


def train_and_evaluate(c: DictConfig):

    # get model and dataset rng seed
    key = jax.random.key(c.seed)
    key, key_model, key_dataset = jax.random.split(key, 3)

    # sharding
    num_fsdp_devices = jax.device_count() // c.num_tp_devices
    if c.get('num_data_devices') is not None:
        num_fsdp_devices = min(c.num_data_devices, num_fsdp_devices)
    mesh = jax.make_mesh((num_fsdp_devices, c.num_tp_devices), ('data', 'model'))
    jax.set_mesh(mesh)
    n_devices = num_fsdp_devices * c.num_tp_devices
    print('sharding mesh:', ', '.join(f'{k}={v}' for k, v in mesh.shape.items()))

    # model
    print('initializing model...')
    c.model.V = int(math.ceil(c.model.V / n_devices) * n_devices) # round V up to enable sharding
    model = model_lib.create_sharded_model(c.model, key_model)
    model_graphdef = nnx.graphdef(model)

    # get num. model parameters
    n_params = {
        'n_param_nonembed': 12 * c.model.L * c.model.D**2,
        'n_param_embed': c.model.D * c.model.V,
        'n_param_actual': utils.get_num_model_params(model),
    }
    for k, v in n_params.items():
        print(f'{k}={v:_}')

    # dataset
    if (c.num_tokens_train is None) and (c.tokens_params_ratio is not None):
        c.num_tokens_train = c.tokens_params_ratio * (n_params['n_param_nonembed'] + n_params['n_param_embed'])
    ds_train, ds_valid = data.load_ds(key_dataset, mesh, c.ds_path, c.model.T, c.opt.microbatch_size, c.num_tokens_valid, c.num_tokens_train)
    if (c.num_tokens_train is None): c.num_tokens_train = ds_train.size

    # optimizer
    num_opt_steps = len(ds_train) // c.opt.grad_acc_steps
    tokens_per_opt_step = c.opt.batch_size * c.model.T
    tx = optimizer_lib.get_optimizer(c.opt, num_opt_steps, tokens_per_opt_step)
    optimizer = optimizer_lib.ModelAndOptimizer(model, tx, stochastic_round=c.opt.stochastic_round, rounding_flip_steps=c.opt.rounding_flip_steps)
    opt_graphdef, opt_state = nnx.split(optimizer)

    # shadow fp32 model: a second model+optimizer trained in fp32 for gradient comparison
    shadow_fp32 = c.opt.get('shadow_fp32', False)
    shadow_opt_graphdef = None
    shadow_opt_state = None
    shadow_model_graphdef = None
    shadow_use_primary_graphdefs = False
    if shadow_fp32:
        from omegaconf import OmegaConf
        print('initializing shadow fp32 model...')
        shadow_model_cfg = OmegaConf.create(OmegaConf.to_container(c.model, resolve=True))
        shadow_model_cfg.param_dtype = 'float32'
        shadow_model_cfg.activ_dtype = 'float32'
        shadow_model = model_lib.create_sharded_model(shadow_model_cfg, key_model)
        shadow_model_graphdef = nnx.graphdef(shadow_model)
        # Overwrite primary model weights with bf16-cast shadow weights so that
        # the primary weights are exactly bf16-rounded versions of the shadow's fp32
        # weights (not independently initialized in bf16, which gives different values).
        shadow_state = nnx.state(shadow_model)
        model_state = nnx.state(model)
        new_model_state = jax.tree.map(
            lambda m, s: jnp.asarray(s).astype(jnp.asarray(m).dtype)
                         if jnp.issubdtype(jnp.asarray(s).dtype, jnp.floating)
                         else m,
            model_state, shadow_state,
        )
        nnx.update(model, new_model_state)
        # Re-split primary model since weights changed
        model_graphdef = nnx.graphdef(model)
        opt_graphdef, opt_state = nnx.split(optimizer)
        shadow_tx = tx
        shadow_optimizer = optimizer_lib.ModelAndOptimizer(
            shadow_model, shadow_tx, stochastic_round=False, rounding_flip_steps=0,
        )
        shadow_opt_graphdef, shadow_opt_state = nnx.split(shadow_optimizer)
        # When primary is also fp32, both models have identical structure.
        # Reuse primary graphdefs for shadow training so XLA compiles one program,
        # giving bitwise-identical updates and keeping cos=1.0 in the fp32 control.
        if c.model.param_dtype == 'float32' and c.model.activ_dtype == 'float32':
            shadow_use_primary_graphdefs = True

    # start wandb
    if jax.process_index() == 0:
        wandb.init(project=c.wandb_project, config=utils.flatten_dict(c), mode=c.wandb_mode, name=c.run_name)
        print(c.run_name)
        wandb.summary.update(n_params)
        wandb.define_metric('train_tokens_seen')
        wandb.define_metric('train_loss', step_metric='train_tokens_seen')
        wandb.define_metric('eval_loss', step_metric='train_tokens_seen')
        wandb.define_metric('zero_update_pct_nonzero_u', step_metric='train_tokens_seen')
        wandb.define_metric('adam_snr', step_metric='train_tokens_seen')
        wandb.define_metric('step_mean_abs_grad', step_metric='train_tokens_seen')
        wandb.define_metric('step_mean_sq_grad', step_metric='train_tokens_seen')
        wandb.define_metric('step_rms_grad', step_metric='train_tokens_seen')
        wandb.define_metric('lost_abs_ratio', step_metric='train_tokens_seen')
        wandb.define_metric('apply_efficiency', step_metric='train_tokens_seen')
        wandb.define_metric('gcast_rel_error', step_metric='train_tokens_seen')
        wandb.define_metric('pcast_rel_error', step_metric='train_tokens_seen')
        wandb.define_metric('gcast_max_rel_error', step_metric='train_tokens_seen')
        wandb.define_metric('pcast_max_rel_error', step_metric='train_tokens_seen')
        wandb.define_metric('pcast_u_weighted_rel_error', step_metric='train_tokens_seen')
        wandb.define_metric('pcast_frac_above_eps', step_metric='train_tokens_seen')
        wandb.define_metric('grad_cos_bf16_fp32', step_metric='train_tokens_seen')
        wandb.define_metric('grad_rel_l2_bf16_fp32', step_metric='train_tokens_seen')
        wandb.define_metric('grad_sign_mismatch_pct', step_metric='train_tokens_seen')
        # log initial validation before any optimizer step
        init_eval_loss = eval_step(opt_state.model, model_graphdef, ds_valid, c.pad_eval)
        init_log = {'eval_loss': init_eval_loss, 'train_tokens_seen': 0}
        if shadow_fp32:
            # Use minimal batch for gradient comparison (must be divisible by data mesh size)
            n_grad = num_fsdp_devices
            init_batch = ds_train[0][:n_grad] if c.opt.grad_acc_steps == 1 else ds_train[0][:n_grad]
            # Use shadow (fp32) graphdef for BOTH so the forward pass runs in fp32 for both.
            grads_p = _compute_grads_tree(opt_state.model, shadow_model_graphdef, init_batch)
            grads_s = _compute_grads_tree(shadow_opt_state.model, shadow_model_graphdef, init_batch)
            init_cos, init_rel_l2, init_sign = compute_grad_diff_metrics(grads_p, grads_s, debug=True)
            init_log['grad_cos_bf16_fp32'] = init_cos
            init_log['grad_rel_l2_bf16_fp32'] = init_rel_l2
            init_log['grad_sign_mismatch_pct'] = init_sign
            print(f'[grad_diff step=0] cos={init_cos:.6f} rel_l2={init_rel_l2:.3e} sign_mismatch={init_sign:.2f}%')
        wandb.log(init_log)

    # training loop
    train_loss_sum, train_loss_num = jnp.zeros([]), 0
    pbar = range(num_opt_steps)
    if jax.process_index() == 0: pbar = tqdm(pbar)
    for step in pbar:

        # get batch
        if c.opt.grad_acc_steps == 1:
            batch = ds_train[step] # [batch_size, T]
        if c.opt.grad_acc_steps > 1:
            batch = ds_train[step*c.opt.grad_acc_steps:(step+1)*c.opt.grad_acc_steps] # [grad_acc_steps, micro_batch_size, T]
            if step == 0 and jax.process_index() == 0:
                print(f"batch.ndim={batch.ndim}")

        do_eval = (train_loss_num + 1) * tokens_per_opt_step >= c.log_every_tokens

        # training step (primary model)
        key, opt_state, batch_loss, update_metrics = train_step(
            key,
            opt_state,
            opt_graphdef,
            model_graphdef,
            batch,
            c.opt.grad_dtype,
            report_zero_update_pct=do_eval,
        )

        # training step (shadow fp32 model) — same batch, fully fp32
        # When primary is also fp32, reuse primary graphdefs so both models
        # get the exact same XLA compilation (avoids FP non-determinism).
        if shadow_fp32:
            s_opt_gd = opt_graphdef if shadow_use_primary_graphdefs else shadow_opt_graphdef
            s_model_gd = model_graphdef if shadow_use_primary_graphdefs else shadow_model_graphdef
            key, shadow_opt_state, _, _ = train_step(
                key,
                shadow_opt_state,
                s_opt_gd,
                s_model_gd,
                batch,
                'float32',
                report_zero_update_pct=False,
            )

        # print dtypes of key tensors after step 10
        if step == 10 and jax.process_index() == 0:
            print('=== dtype report at step 10 ===')
            def _get_float_dtypes(tree):
                dtypes = set()
                for x in jax.tree.leaves(tree):
                    try:
                        arr = jnp.asarray(x)
                        if jnp.issubdtype(arr.dtype, jnp.floating):
                            dtypes.add(arr.dtype)
                    except Exception:
                        pass
                return dtypes
            print(f'  model param dtypes:     {_get_float_dtypes(opt_state.model)}')
            print(f'  optimizer state dtypes: {_get_float_dtypes(opt_state.opt_state)}')
            print('================================')

        # logging
        train_loss_sum += batch_loss
        train_loss_num += 1
        if do_eval:
            eval_loss = eval_step(opt_state.model, model_graphdef, ds_valid, c.pad_eval)
            metrics = {}
            metrics['train_loss'] = train_loss_sum / train_loss_num
            metrics['eval_loss'] = eval_loss
            metrics['train_tokens_seen'] = (step+1) * tokens_per_opt_step
            metrics['zero_update_pct_nonzero_u'] = float(update_metrics[0])
            metrics['adam_snr'] = float(update_metrics[1])
            metrics['step_mean_abs_grad'] = float(update_metrics[2])
            metrics['step_mean_sq_grad'] = float(update_metrics[3])
            metrics['step_rms_grad'] = float(update_metrics[4])
            metrics['lost_abs_ratio'] = float(update_metrics[5])
            metrics['apply_efficiency'] = float(update_metrics[6])
            metrics['gcast_rel_error'] = float(update_metrics[7])
            metrics['pcast_rel_error'] = float(update_metrics[8])
            metrics['gcast_max_rel_error'] = float(update_metrics[9])
            metrics['pcast_max_rel_error'] = float(update_metrics[10])
            metrics['pcast_u_weighted_rel_error'] = float(update_metrics[11])
            metrics['pcast_frac_above_eps'] = float(update_metrics[12])

            if shadow_fp32:
                grad_batch = batch[:num_fsdp_devices] if batch.ndim == 2 else batch[0][:num_fsdp_devices]
                grads_p = _compute_grads_tree(opt_state.model, shadow_model_graphdef, grad_batch)
                grads_s = _compute_grads_tree(shadow_opt_state.model, shadow_model_graphdef, grad_batch)
                grad_cos, grad_rel_l2, grad_sign_pct = compute_grad_diff_metrics(grads_p, grads_s)
                metrics['grad_cos_bf16_fp32'] = grad_cos
                metrics['grad_rel_l2_bf16_fp32'] = grad_rel_l2
                metrics['grad_sign_mismatch_pct'] = grad_sign_pct
                if jax.process_index() == 0:
                    print(f'[grad_diff] step={step} cos={grad_cos:.4f} rel_l2={grad_rel_l2:.3e} sign_mismatch={grad_sign_pct:.2f}%')

            if jax.process_index() == 0:
                wandb.log(metrics)
                pbar.set_postfix_str(f'train={metrics["train_loss"]:.2f}, eval={metrics["eval_loss"]:.2f}')
            train_loss_sum, train_loss_num = jnp.zeros([]), 0

    # eval at end of training
    eval_loss = eval_step(opt_state.model, model_graphdef, ds_valid, c.pad_eval)
    if jax.process_index() == 0:
        wandb.log({
            'eval_loss': eval_loss,
            'train_tokens_seen': num_opt_steps * tokens_per_opt_step,
        })
        wandb.finish()
