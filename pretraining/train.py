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


def _compute_num_opt_steps(c: DictConfig):
    base_bs = int(c.opt.batch_size)
    initial_bs = c.opt.get('initial_batch_size')
    initial_tokens = int(c.opt.get('initial_batch_tokens', 0) or 0)
    if initial_bs is None or initial_tokens <= 0:
        switch_bs = c.opt.get('switch_batch_size')
        switch_tokens = int(c.opt.get('switch_at_tokens', 0) or 0)
        if switch_bs is None or switch_tokens <= 0:
            return int(c.num_tokens_train // (base_bs * c.model.T))

        switch_bs = int(switch_bs)
        switch_tokens = min(switch_tokens, int(c.num_tokens_train))
        initial_steps = switch_tokens // (base_bs * c.model.T)
        tokens_after_switch = int(c.num_tokens_train) - initial_steps * base_bs * c.model.T
        later_steps = tokens_after_switch // (switch_bs * c.model.T)
        return int(initial_steps + later_steps)

    initial_bs = int(initial_bs)
    initial_tokens = min(initial_tokens, int(c.num_tokens_train))
    initial_steps = initial_tokens // (initial_bs * c.model.T)
    tokens_after_initial = int(c.num_tokens_train) - initial_steps * initial_bs * c.model.T
    later_steps = tokens_after_initial // (base_bs * c.model.T)
    return int(initial_steps + later_steps)


def _sample_perturbed_model_state(key, model_state, std_frac: float):
    keys = otu.tree_split_key_like(key, model_state)

    def _perturb_leaf(leaf_key, p):
        p_arr = jnp.asarray(p)
        if not jnp.issubdtype(p_arr.dtype, jnp.floating):
            return p
        p_f32 = p_arr.astype(jnp.float32)
        rms = jnp.sqrt(jnp.mean(jnp.square(p_f32))) + jnp.asarray(1e-8, dtype=jnp.float32)
        noise = jax.random.normal(leaf_key, p_f32.shape, dtype=jnp.float32) * (std_frac * rms)
        return (p_f32 + noise).astype(p_arr.dtype)

    return jax.tree.map(_perturb_leaf, keys, model_state)


@partial(jax.jit, static_argnames=('model_graphdef', 'pad'))
def loss_fn(model_state, model_graphdef, x, pad=False): # [B, T]
    model = nnx.merge(model_graphdef, model_state)
    y = jnp.roll(x, -1, axis=1)
    loss_mask = data.pad_mask(x) if pad else jnp.ones(x.shape, dtype=bool)
    loss_mask = loss_mask.at[:, -1].set(False)
    logits = model(x) # [B, T, V]
    losses = optax.softmax_cross_entropy_with_integer_labels(logits.astype(jnp.float32), y) # [B, T]
    return (losses * loss_mask).sum() / loss_mask.sum()


@partial(jax.jit, static_argnames=('opt_graphdef', 'model_graphdef', 'grad_dtype', 'report_update_metrics'))
def train_step(key, opt_state, opt_graphdef, model_graphdef, batch, grad_dtype=None, report_update_metrics: bool = False):
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
        report_update_metrics=report_update_metrics,
    )
    opt_state = nnx.state(optimizer)
    return key, opt_state, loss, update_metrics


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
    data_batch_size = c.opt.microbatch_size
    phase_data_batch_size = None
    switch_data_batch_size = None
    if (c.opt.get('initial_batch_size') is not None) and int(c.opt.get('initial_batch_tokens', 0) or 0) > 0:
        initial_bs = int(c.opt.initial_batch_size)
        initial_phase_cap = c.opt.get('initial_phase_max_microbatch_size')
        if initial_phase_cap is None:
            phase_data_batch_size = initial_bs
        else:
            phase_data_batch_size = min(initial_bs, int(initial_phase_cap))
        data_batch_size = 1
    elif (c.opt.get('switch_batch_size') is not None) and int(c.opt.get('switch_at_tokens', 0) or 0) > 0:
        switch_bs = int(c.opt.switch_batch_size)
        switch_phase_cap = c.opt.get('switch_phase_max_microbatch_size')
        if switch_phase_cap is None:
            switch_data_batch_size = switch_bs
        else:
            switch_data_batch_size = min(switch_bs, int(switch_phase_cap))

    if (c.opt.get('initial_batch_size') is not None) and int(c.opt.get('initial_batch_tokens', 0) or 0) > 0:
        key_dataset_bs1, key_dataset_phase = jax.random.split(key_dataset)
        ds_train, ds_valid = data.load_ds(key_dataset_bs1, mesh, c.ds_path, c.model.T, data_batch_size, c.num_tokens_valid, c.num_tokens_train)
        ds_train_initial, _ = data.load_ds(
            key_dataset_phase,
            mesh,
            c.ds_path,
            c.model.T,
            int(phase_data_batch_size),
            c.num_tokens_valid,
            c.num_tokens_train,
        )
        ds_train_switch = None
    elif (c.opt.get('switch_batch_size') is not None) and int(c.opt.get('switch_at_tokens', 0) or 0) > 0:
        ds_train, ds_valid = data.load_ds(key_dataset, mesh, c.ds_path, c.model.T, data_batch_size, c.num_tokens_valid, c.num_tokens_train)
        ds_train_switch, _ = data.load_ds(
            key_dataset,
            mesh,
            c.ds_path,
            c.model.T,
            int(switch_data_batch_size),
            c.num_tokens_valid,
            c.num_tokens_train,
        )
        ds_train_initial = None
    else:
        ds_train, ds_valid = data.load_ds(key_dataset, mesh, c.ds_path, c.model.T, data_batch_size, c.num_tokens_valid, c.num_tokens_train)
        ds_train_initial = None
        ds_train_switch = None
    if (c.num_tokens_train is None): c.num_tokens_train = ds_train.size

    # optimizer
    num_opt_steps = _compute_num_opt_steps(c)
    base_tokens_per_opt_step = c.opt.batch_size * c.model.T
    tx, lr_schedule = optimizer_lib.get_optimizer(c.opt, num_opt_steps, base_tokens_per_opt_step, c.num_tokens_train)
    optimizer = optimizer_lib.ModelAndOptimizer(
        model,
        tx,
        stochastic_round=c.opt.stochastic_round,
        rounding_flip_steps=c.opt.rounding_flip_steps,
        lr_schedule=lr_schedule,
    )
    opt_graphdef, opt_state = nnx.split(optimizer)

    # start wandb
    if jax.process_index() == 0:
        wandb.init(project=c.wandb_project, config=utils.flatten_dict(c), mode=c.wandb_mode, name=c.run_name)
        print(c.run_name)
        wandb.summary.update(n_params)
        wandb.define_metric('train_tokens_seen')
        wandb.define_metric('train_loss', step_metric='train_tokens_seen')
        wandb.define_metric('eval_loss', step_metric='train_tokens_seen')
        wandb.define_metric('update_cos_pre_post_cast', step_metric='train_tokens_seen')
        wandb.define_metric('cast_error_mean', step_metric='train_tokens_seen')
        wandb.define_metric('cast_bias_along_update', step_metric='train_tokens_seen')
        wandb.define_metric('cast_error_rel_norm', step_metric='train_tokens_seen')
        wandb.define_metric('update_cos_pre_post_cast_nolr', step_metric='train_tokens_seen')
        wandb.define_metric('cast_bias_along_update_nolr', step_metric='train_tokens_seen')
        # log initial validation before any optimizer step
        init_eval_loss = eval_step(opt_state.model, model_graphdef, ds_valid, c.pad_eval)
        init_log = {'eval_loss': init_eval_loss, 'train_tokens_seen': 0}
        wandb.log(init_log)

    # training loop
    train_loss_sum, train_loss_num = jnp.zeros([]), 0
    train_tokens_since_log = 0
    train_tokens_seen = 0
    perturb_done = False
    data_cursor = 0
    pbar = range(num_opt_steps)
    if jax.process_index() == 0: pbar = tqdm(pbar)
    last_update_log_metrics = None
    for step in pbar:
        if train_tokens_seen >= c.num_tokens_train:
            break

        current_batch_size = int(c.opt.batch_size)
        if (
            c.opt.get('initial_batch_size') is not None
            and int(c.opt.get('initial_batch_tokens', 0) or 0) > 0
            and train_tokens_seen < int(c.opt.initial_batch_tokens)
        ):
            current_batch_size = int(c.opt.initial_batch_size)
        elif (
            c.opt.get('switch_batch_size') is not None
            and int(c.opt.get('switch_at_tokens', 0) or 0) > 0
            and train_tokens_seen >= int(c.opt.switch_at_tokens)
        ):
            current_batch_size = int(c.opt.switch_batch_size)
        current_grad_acc_steps = max(1, current_batch_size // data_batch_size)
        tokens_this_step = current_batch_size * c.model.T

        # get batch
        using_initial_phase = (
            ds_train_initial is not None
            and c.opt.get('initial_batch_size') is not None
            and int(c.opt.get('initial_batch_tokens', 0) or 0) > 0
            and train_tokens_seen < int(c.opt.initial_batch_tokens)
        )
        using_switch_phase = (
            ds_train_switch is not None
            and c.opt.get('switch_batch_size') is not None
            and int(c.opt.get('switch_at_tokens', 0) or 0) > 0
            and train_tokens_seen >= int(c.opt.switch_at_tokens)
        )

        if using_initial_phase:
            phase_step = train_tokens_seen // tokens_this_step
            phase_grad_acc_steps = max(1, current_batch_size // int(phase_data_batch_size))
            phase_start = int(phase_step) * phase_grad_acc_steps
            if phase_grad_acc_steps == 1:
                batch = ds_train_initial[phase_start]
            else:
                batch = ds_train_initial[phase_start:phase_start+phase_grad_acc_steps]
        elif using_switch_phase:
            switch_step = train_tokens_seen // tokens_this_step
            switch_grad_acc_steps = max(1, current_batch_size // int(switch_data_batch_size))
            switch_start = int(switch_step) * switch_grad_acc_steps
            if switch_grad_acc_steps == 1:
                batch = ds_train_switch[switch_start]
            else:
                batch = ds_train_switch[switch_start:switch_start+switch_grad_acc_steps]
        elif current_grad_acc_steps == 1:
            batch = ds_train[data_cursor] # [batch_size, T]
            data_cursor += 1
        elif current_grad_acc_steps > 1:
            batch = ds_train[data_cursor:data_cursor+current_grad_acc_steps] # [grad_acc_steps, micro_batch_size, T]
            data_cursor += current_grad_acc_steps
            if step == 0 and jax.process_index() == 0:
                print(f"batch.ndim={batch.ndim}")

        do_eval = train_tokens_since_log + tokens_this_step >= c.log_every_tokens
        log_update_metrics = do_eval or step < c.get('update_cos_first_n_steps', 0)

        # training step (primary model)
        key, opt_state, batch_loss, update_metrics = train_step(
            key,
            opt_state,
            opt_graphdef,
            model_graphdef,
            batch,
            c.opt.grad_dtype,
            report_update_metrics=log_update_metrics,
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
        train_tokens_seen += tokens_this_step
        train_tokens_since_log += tokens_this_step

        if (
            (not perturb_done)
            and float(c.opt.get('perturb_std_frac', 0.0) or 0.0) > 0.0
            and int(c.opt.get('perturb_at_tokens', 0) or 0) > 0
            and train_tokens_seen >= int(c.opt.perturb_at_tokens)
        ):
            key, key_perturb = jax.random.split(key)
            optimizer = nnx.merge(opt_graphdef, opt_state)
            perturbed_model = _sample_perturbed_model_state(key_perturb, opt_state.model, float(c.opt.perturb_std_frac))
            nnx.update(optimizer.model, perturbed_model)
            opt_state = nnx.state(optimizer)
            perturb_done = True
            if jax.process_index() == 0:
                print(
                    f"[perturb] train_tokens_seen={train_tokens_seen} "
                    f"std_frac={float(c.opt.perturb_std_frac):.3e}"
                )
                if c.wandb_mode != 'disabled':
                    wandb.log({'perturb_std_frac': float(c.opt.perturb_std_frac), 'train_tokens_seen': train_tokens_seen})

        if do_eval:
            eval_loss = eval_step(opt_state.model, model_graphdef, ds_valid, c.pad_eval)
            metrics = {}
            metrics['train_loss'] = train_loss_sum / train_loss_num
            metrics['eval_loss'] = eval_loss
            metrics['train_tokens_seen'] = train_tokens_seen
            if log_update_metrics:
                metrics['update_cos_pre_post_cast'] = float(update_metrics[0])
                metrics['cast_error_mean'] = float(update_metrics[1])
                metrics['cast_bias_along_update'] = float(update_metrics[2])
                metrics['cast_error_rel_norm'] = float(update_metrics[3])
                metrics['update_cos_pre_post_cast_nolr'] = float(update_metrics[4])
                metrics['cast_bias_along_update_nolr'] = float(update_metrics[5])
                last_update_log_metrics = {
                    'update_cos_pre_post_cast': metrics['update_cos_pre_post_cast'],
                    'cast_error_mean': metrics['cast_error_mean'],
                    'cast_bias_along_update': metrics['cast_bias_along_update'],
                    'cast_error_rel_norm': metrics['cast_error_rel_norm'],
                    'update_cos_pre_post_cast_nolr': metrics['update_cos_pre_post_cast_nolr'],
                    'cast_bias_along_update_nolr': metrics['cast_bias_along_update_nolr'],
                }

            if jax.process_index() == 0:
                wandb.log(metrics)
                pbar.set_postfix_str(f'train={metrics["train_loss"]:.2f}, eval={metrics["eval_loss"]:.2f}')
            train_loss_sum, train_loss_num = jnp.zeros([]), 0
            train_tokens_since_log = 0
        elif log_update_metrics:
            metrics = {
                'update_cos_pre_post_cast': float(update_metrics[0]),
                'cast_error_mean': float(update_metrics[1]),
                'cast_bias_along_update': float(update_metrics[2]),
                'cast_error_rel_norm': float(update_metrics[3]),
                'update_cos_pre_post_cast_nolr': float(update_metrics[4]),
                'cast_bias_along_update_nolr': float(update_metrics[5]),
                'train_tokens_seen': train_tokens_seen,
            }
            last_update_log_metrics = {
                'update_cos_pre_post_cast': metrics['update_cos_pre_post_cast'],
                'cast_error_mean': metrics['cast_error_mean'],
                'cast_bias_along_update': metrics['cast_bias_along_update'],
                'cast_error_rel_norm': metrics['cast_error_rel_norm'],
                'update_cos_pre_post_cast_nolr': metrics['update_cos_pre_post_cast_nolr'],
                'cast_bias_along_update_nolr': metrics['cast_bias_along_update_nolr'],
            }
            if jax.process_index() == 0:
                wandb.log(metrics)

    # eval at end of training
    eval_loss = eval_step(opt_state.model, model_graphdef, ds_valid, c.pad_eval)
    if jax.process_index() == 0:
        final_metrics = {
            'eval_loss': eval_loss,
            'train_tokens_seen': train_tokens_seen,
        }
        if last_update_log_metrics is not None:
            final_metrics.update(last_update_log_metrics)
        wandb.log(final_metrics)
        wandb.finish()
