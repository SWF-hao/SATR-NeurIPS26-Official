import jax
import jax.numpy as jnp
import flax.linen as nn


def conn_dense(kernel, x):
    # Check dtypes
    assert kernel.dtype == jnp.bool_, "Kernel must be boolean."
    assert x.dtype != jnp.bool_, "Inputs must not be boolean."
    
    # matmul
    return jax.lax.dot_general(x, kernel.astype(x.dtype), (((x.ndim - 1,), (0,)), ((), ())))


class DenseRNN(nn.Module):
    """Dense RNN network - ANN version with Bernoulli connections like SNN"""
    
    out_dims: int
    hidden_dims: int = 256
    activation: str = "tanh"
    expected_sparsity: float = 0.5
    
    # Same parameters as SNN
    K_in: float = 0.1
    K_h: float = 1.0
    K_out: float = 5.0
    
    @nn.compact
    def __call__(self, carry, input):
        # Get activation function
        if self.activation == "tanh":
            activation_fn = jax.nn.tanh
        elif self.activation == "relu":
            activation_fn = jax.nn.relu
        elif self.activation == "sigmoid":
            activation_fn = jax.nn.sigmoid
        else:
            activation_fn = jax.nn.tanh
        
        # Boolean connection matrices (same as SNN)
        in_dims = input.shape[-1]
        kernel_in = self.param("kernel_in", nn.initializers.zeros, (2 * in_dims, self.hidden_dims), jnp.bool_)
        kernel_h = self.param("kernel_h", nn.initializers.zeros, (self.hidden_dims, self.hidden_dims), jnp.bool_)
        kernel_out = self.param("kernel_out", nn.initializers.zeros, (self.hidden_dims, self.out_dims), jnp.bool_)
        
        # Weight scaling (same as SNN)
        R_in = self.K_in * jnp.sqrt(2 / (self.expected_sparsity * in_dims))
        R = self.K_h * jnp.sqrt(2 / (self.expected_sparsity * self.hidden_dims))
        R_out = self.K_out * jnp.sqrt(1 / (self.expected_sparsity * self.hidden_dims))
        
        # Input processing (same as SNN)
        x = input.astype(jnp.float32)
        i_in = R_in * conn_dense(kernel_in, jnp.concatenate([x, -x], axis=-1))
        
        # RNN state
        if carry is None:
            h = jnp.zeros((input.shape[0], self.hidden_dims))
        else:
            h = carry
        
        # RNN computation
        hidden_proj = R * conn_dense(kernel_h, h)
        new_h = activation_fn(i_in + hidden_proj)
        
        # Output
        output = R_out * conn_dense(kernel_out, new_h)
        
        return new_h, output
    
    def initial_carry(self, key, batch_size):
        return jnp.zeros((batch_size, self.hidden_dims))
