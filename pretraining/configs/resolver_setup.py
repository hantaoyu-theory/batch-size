import operator as op
from omegaconf import OmegaConf

OmegaConf.register_new_resolver('floordiv', op.floordiv)
OmegaConf.register_new_resolver('mul', op.mul)
OmegaConf.register_new_resolver('min', min)
OmegaConf.register_new_resolver('pow', pow)

_DTYPE_SHORT = {
    'float32': 'fp32',
    'float16': 'fp16',
    'bfloat16': 'bf16',
    'float8_e4m3fn': 'fp8_e4m3fn',
    'float8_e5m2': 'fp8_e5m2',
}
OmegaConf.register_new_resolver('short_dtype', lambda d: _DTYPE_SHORT.get(d, d))