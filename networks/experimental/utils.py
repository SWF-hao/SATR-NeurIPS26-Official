import jax.numpy as jnp

from jax._src.random import _random_bits, _check_prng_key

MAX_INT = {
    jnp.int32 : 0x7fffffff,
    jnp.int16 : 0x7fff,
    jnp.int8 : 0x7f,
}
    
def int_max(dtype: jnp.dtype) -> int:
    
    return MAX_INT[dtype]

def generate_bernoulli(key, p, batch_dims=(), rng_bits=16):

    key, _ = _check_prng_key(key)

    rng = _random_bits(key, rng_bits, batch_dims + p.shape)
    p = (((1 << rng_bits) - 1) * p).astype(rng.dtype)

    return rng < p