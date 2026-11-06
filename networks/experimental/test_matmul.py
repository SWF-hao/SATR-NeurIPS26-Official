import time

import jax
import jax.numpy as jnp
from jax._src.random import _random_bits, _check_prng_key

jax.config.update("jax_default_prng_impl", "unsafe_rbg")


def bitset_generate_bernoulli(key, p, bitset_dtype, batch_dims=(), a_bits=16):
    """Generate bernoulli distribution"""
    key, _ = _check_prng_key(key)

    a = _random_bits(key, a_bits, batch_dims + p.shape)
    p = (((1 << a_bits) - 1) * p).astype(a.dtype)

    return bitset_from_bool(a < p, bitset_dtype)


def bitset_from_bool(x, bitset_dtype):
    """Create bitset from boolean (on last dimension)."""
    assert (x.shape[-1] % jnp.iinfo(bitset_dtype).bits) == 0, "Array length must be multiple of bits of bitset dtype."

    # To destination dtype
    x = x.view(bitset_dtype)

    # Bitwise or
    bits = jnp.arange(8, dtype=jnp.uint8)
    x    = x.reshape(x.shape[:-1] + (x.shape[-1] // 8, 8))
    x    = jax.lax.reduce(x << bits, 0, jax.lax.bitwise_or, (x.ndim - 1, ))
    return x


def bitset_matvec_mul(A, b, out_dtype):
    """Bitset matrix-vector multiplication Y = Ab (on last dimension).
    A: ... x N x (M/Q) matrix
    B: ... x (M/Q)     vector 
    Y: ... x N         vector"""
    assert A.dtype == b.dtype, "A and b must have the same dtype."
    assert A.shape[-1] == b.shape[-1], "The last dimension of A and b must be of same size."

    # Broadcasted bitwise and + popcount
    x = jax.lax.bitwise_and(A, jnp.expand_dims(b, -2))
    # Popcount
    x = jax.lax.population_count(x)
    # Accumulate (reduce sum)
    return jax.lax.reduce(x.astype(out_dtype), 0, jax.lax.add, (x.ndim - 1, ))

def bool_matvec_mul(A, b, bitset_dtype = jnp.uint32, out_dtype = jnp.uint8):

    assert A.dtype == jnp.bool_
    assert b.dtype == jnp.bool_
    A = bitset_from_bool(A, bitset_dtype)
    b = bitset_from_bool(b, bitset_dtype)

    return bitset_matvec_mul(A, b, out_dtype)


def test_bmm(
    key,

    batch_size: int = 10240,
    mat_size:   int = 256,
    length:     int = 2000,

    accumulate_type = jnp.uint8,
    bitset_dtype = jnp.uint32,
):
    # generate matrix
    key_mat, key_data = jax.random.split(key)
    mat = jax.random.bernoulli(key_mat, 0.5, (batch_size, mat_size, mat_size))

    # scan matmul
    # prng_key = _check_prng_key(key_mat)[0]

    def _forward_bitset(_carry, input):

        # y = bool_matvec_mul(mat, _carry, bitset_dtype = bitset_dtype, out_dtype = accumulate_type)
        b = bitset_from_bool(_carry, bitset_dtype)

        res =  bitset_matvec_mul(set_mat, b, accumulate_type)
        
        return res.astype(_carry.dtype), None

    def _forward(_carry, _input):
        y = jax.lax.dot_general(_carry, mat, (((_carry.ndim - 1,), (1,)), ((0,), (0,))), preferred_element_type=accumulate_type)

        # y = jnp.squeeze(jnp.expand_dims(_carry.astype(jnp.float32), axis=-2) @ mat, axis=-2)
        return y.astype(_carry.dtype), None

    x = jax.random.bernoulli(key_data, 0.5, (batch_size, mat_size))

    y1 = jnp.squeeze(jnp.matmul(mat.astype(accumulate_type), x.reshape(batch_size, mat_size, 1).astype(accumulate_type)))
    y2 = bool_matvec_mul(mat, x, bitset_dtype = bitset_dtype, out_dtype = accumulate_type)
    assert jnp.array_equal(y1, y2)

    start_time = time.time()
    
    x, _ = jax.lax.scan(_forward, x, None, length)
    elapsed = time.time() - start_time
    print(f"Normal {elapsed:.2f}")
    
    start_time = time.time()
    set_mat = bitset_from_bool(mat, bitset_dtype = bitset_dtype)
    x, _ = jax.lax.scan(_forward_bitset, x, None, length)
    
    elapsed = time.time() - start_time
    print(f"Bitset {elapsed:.2f}")

    return x


def main():
    key = jax.random.PRNGKey(0)

    test_bmm(key)


if __name__ == "__main__":
    main()