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


@partial(jax.jit, static_argnames=('opt_graphdef', 'model_graphdef', 'grad_dtype'), donate_argnames=('opt_state'))
def train_step(key, opt_state, opt_graphdef, model_graphdef, batch, grad_dtype=None):
    key, key_opt = jax.random.split(key)

    # compute grads from a single micro-batch
    if batch.ndim == 2:

        if grad_dtype == "float32":
            params_for_grad = jax.tree.map(lambda p: p.astype(grad_dtype), opt_state.model)
            loss, grads = jax.value_and_grad(loss_fn)(params_for_grad, model_graphdef, batch)
        else:
            loss, grads = jax.value_and_grad(loss_fn)(opt_state.model, model_graphdef, batch)

    # compute grads from multiple micro-batches (using gradient accumulation)
    if batch.ndim == 3:
        raise ValueError("gradient accumulation (batch.ndim==3) is not supported")
        loss = 0
        grads = otu.tree_zeros_like(opt_state.model, dtype=jnp.float32)
        def step_fn(i , args):
            loss, grads = args
            batch_loss, batch_grads = jax.value_and_grad(loss_fn)(opt_state.model, model_graphdef, batch[i])
            loss = (i*loss + batch_loss) / (i+1)
            grads = jax.tree.map(lambda m, g: (i*m + g) / (i+1), grads, batch_grads)
            return loss, grads
        loss, grads = jax.lax.fori_loop(0, len(batch), step_fn, (loss, grads))

    optimizer = nnx.merge(opt_graphdef, opt_state)
    optimizer.update(key_opt, grads, grad_dtype=grad_dtype)
    opt_state = nnx.state(optimizer)
    return key, opt_state, loss


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
    optimizer = optimizer_lib.ModelAndOptimizer(model, tx, stochastic_round=c.opt.stochastic_round)
    opt_graphdef, opt_state = nnx.split(optimizer)

    # start wandb
    if jax.process_index() == 0:
        wandb.init(project=c.wandb_project, config=utils.flatten_dict(c), mode=c.wandb_mode, name=c.run_name)
        print(c.run_name)
        wandb.summary.update(n_params)
        wandb.define_metric('train_tokens_seen')
        wandb.define_metric('train_loss', step_metric='train_tokens_seen')
        wandb.define_metric('eval_loss', step_metric='train_tokens_seen')
        # log initial validation before any optimizer step
        init_eval_loss = eval_step(opt_state.model, model_graphdef, ds_valid, c.pad_eval)
        wandb.log({
            'eval_loss': init_eval_loss,
            'train_tokens_seen': 0,
        })

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

        # training step
        key, opt_state, batch_loss = train_step(key, opt_state, opt_graphdef, model_graphdef, batch, c.opt.grad_dtype)

        # logging
        train_loss_sum += batch_loss
        train_loss_num += 1
        if train_loss_num * tokens_per_opt_step >= c.log_every_tokens:
            eval_loss = eval_step(opt_state.model, model_graphdef, ds_valid, c.pad_eval)
            metrics = {}
            metrics['train_loss'] = train_loss_sum / train_loss_num
            metrics['eval_loss'] = eval_loss
            metrics['train_tokens_seen'] = (step+1) * tokens_per_opt_step
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
