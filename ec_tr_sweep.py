import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
from functools import partial
from typing import Any, Dict, Tuple, Optional
import time
import builtins
import dataclasses

import jax
import jax.numpy as jnp

import flax
import optax

from brax import envs
from brax.training.acme import running_statistics
from brax.training.acme import specs

from omegaconf import OmegaConf
from tqdm import tqdm
import wandb
import optuna

from optuna.samplers import GridSampler
from networks import NETWORKS
from utils.functions import mean_weight_abs, finitemean, save_obj_to_file

# bitset optimization
from networks.experimental import bitset_from_bool
from networks.experimental.utils import generate_bernoulli

# histogram logging deps
import numpy as np


# Use RBG generator for less memory consumption
# Default RNG needs 2*N extra memory, while RBG needs none, when generating array with size N
# https://jax.readthedocs.io/en/latest/jax.random.html
# jax.config.update("jax_default_prng_impl", "unsafe_rbg")

# Hack for resolving bfloat16 pickling issue https://github.com/google/jax/issues/8505
builtins.bfloat16 = jnp.dtype("bfloat16").type


def init_wandb_safe(project=None, name=None, group=None, config=None, max_retries=3, timeout=300):
    """
    Safely initialize wandb with timeout settings and retry mechanism
    
    Args:
        project: wandb project name
        name: run name
        group: run group name
        config: configuration dictionary
        max_retries: maximum number of retries
        timeout: initialization timeout in seconds, default 300 seconds
    """
    for attempt in range(max_retries):
        try:
            init_kwargs = {
                "reinit": True,
                "project": project,
                "name": name,
                "config": config,
                "settings": wandb.Settings(init_timeout=timeout)
            }
            if group:
                init_kwargs["group"] = group
            
            wandb.init(**init_kwargs)
            print(f"✅ wandb initialized successfully (attempt {attempt + 1}/{max_retries})")
            return
        except wandb.errors.CommError as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 10  # Incremental wait time
                print(f"⚠️ wandb initialization timeout (attempt {attempt + 1}/{max_retries}), retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"❌ wandb initialization failed, reached maximum retries. Trying offline mode...")
                try:
                    # Try offline mode
                    init_kwargs = {
                        "reinit": True,
                        "project": project,
                        "name": name,
                        "config": config,
                        "mode": "offline"
                    }
                    if group:
                        init_kwargs["group"] = group
                    wandb.init(**init_kwargs)
                    print("✅ wandb offline mode initialized successfully")
                    return
                except Exception as offline_error:
                    print(f"❌ wandb offline mode also failed: {offline_error}")
                    raise e
        except Exception as e:
            print(f"❌ wandb initialization error: {e}")
            raise


@flax.struct.dataclass
class ESConfig:
    # Network, optim & env class
    network_cls: Any = None
    optim_cls:   Any = None
    env_cls:     Any = None
    eval_env_cls: Any = None  # Separate environment for evaluation (can be None to use env_cls)

    # [Hyperparameters] ES
    pop_size:       int = 10240
    lr:           float = 0.15
    # lr:           float = 0.5# for lr is z-score transform, should be larger than 0.15


    eps:          float = 1e-3

    weight_decay: float = 0.    # For sparsity regularization
    
    # SGD momentum parameter
    # Momentum is used to smooth updates from custom update formula
    # Formula: v_t = momentum * v_{t-1} + custom_update
    # where custom_update = sign(grad) * sqrt(2*lr*(1-p)*p)
    # 
    # Recommended values:
    #   - 0.0: do not use momentum, directly use custom update
    #   - 0.9: common value, provides moderate smoothing and acceleration
    #   - 0.95: stronger momentum, smoother but may respond slower
    #   - 0.99: very strong momentum, mainly for very stable training
    momentum:     float = 0.9   # Momentum coefficient for SGD (0.0 to disable)

    # [Hyperparameters] Warmup
    warmup_steps:   int = 0

    # [Hyperparameters] Eval
    eval_size:      int = 128

    # [Computing] Data types
    action_dtype: Any   = jnp.float32  # brax uses fp32

    p_dtype:       Any  = jnp.float32
    network_dtype: Any  = jnp.float32
    bitset_dtype:  jnp.dtype = jnp.uint32  # For bitset optimization
    # Fitness transform: centered_rank or z_score
    transform_type: str = "centered_rank"
    # Update method: "scale_delta_with_abs", "mtr_rsrp", "ba_rsrp", or "spherical_trust_region"
    # update_method: str = "scale_delta_with_abs"
    update_method: str = "spherical_trust_region"
    # Experiment option: scale delta_preclip by ln(abs_param) before logging
    scale_delta_with_abs: bool = False
    scale_factor: float = 0.5
    # Metaplastic Trust-Region RSRP parameters
    metaplastic_alpha: float = 0.3
    # Boundary-aware RSRP parameters
    ba_alpha: float = 0.5  # Exponent for same-direction updates
    ba_beta: float = 0.5   # Exponent for opposite-direction updates
    # Use reference reward for centered_rank transform
    use_reward_ref: bool = False
    # Dynamic sampling: skip update if all rewards < ref_reward
    dynamic_sampling: bool = False
    # TRPO-style update parameters
    # TRPO (Trust Region Policy Optimization) uses natural gradient direction with fixed KL constraint
    # This provides more stable updates compared to standard optimizers
    # 
    # Usage: Set use_trpo_update=True in config to enable
    # Recommended values:
    #   - trpo_kl_eps: 0.01 (smaller = more conservative updates, larger = more aggressive)
    #   - trpo_damping: 1e-8 (numerical stability, usually keep default)
    #   - trpo_alpha_max: None or float (maximum step size, None = no limit)
    use_trpo_update: bool = False  # Enable TRPO-style natural gradient update
    trpo_kl_eps: float = 0.01  # Fixed KL divergence constraint (trust region radius)
    trpo_damping: float = 1e-8  # Numerical stability term for KL constraint
    trpo_alpha_max: Optional[float] = None  # Maximum step size (None = no limit, recommended: 1.0-2.0)

@flax.struct.dataclass
class RunnerState:
    key: Any
    # Normalizer
    normalizer_state: running_statistics.RunningStatisticsState
    # Env reset state pool
    env_reset_pool: Any
    # Network optimization
    params:        Any
    fixed_weights: Any
    opt_state:     Any
    # Separate reset pool for evaluation environment
    eval_env_reset_pool: Any = None
    # Reference reward for reward ref transform
    ref_reward:    Any = None
    # Total environment steps from training start
    total_env_steps: jnp.ndarray = jnp.array(0, dtype=jnp.int32)
    # Momentum state for custom update (only used if momentum > 0)
    momentum_state: Any = None


@flax.struct.dataclass
class PopulationState:
    # Network
    network_params: Any
    network_states: Any
    # Env
    env_states:     Any
    # Fitness
    fitness_totrew: jnp.ndarray
    fitness_sum:    jnp.ndarray
    fitness_n:      jnp.ndarray
    # Environment steps counter for current generation
    env_steps:      jnp.ndarray = jnp.array(0, dtype=jnp.int32)


def _centered_rank_transform(x: jnp.ndarray) -> jnp.ndarray:
    """Centered rank from: https://arxiv.org/pdf/1703.03864.pdf"""

    shape = x.shape
    x     = x.ravel()

    x = jnp.argsort(jnp.argsort(x))
    x = x / (len(x) - 1) - .5
    return x.reshape(shape)


def _centered_rank_transform_with_ref(x: jnp.ndarray, ref_reward: jnp.ndarray) -> jnp.ndarray:
    """Centered rank with reference reward alignment.
    
    Args:
        x: Fitness values to transform
        ref_reward: Reference reward value to align with
    
    Returns:
        Transformed weights aligned to reference reward position
    """
    shape = x.shape
    x_flat = x.ravel()
    
    # Find the closest element to ref_reward and replace it
    distances = jnp.abs(x_flat - ref_reward)
    closest_idx = jnp.argmin(distances)
    x_aligned = x_flat.at[closest_idx].set(ref_reward)
    
    # Compute ranking on aligned array
    ranks = jnp.argsort(jnp.argsort(x_aligned))
    
    # Find the rank of ref_reward
    ref_rank = ranks[closest_idx]
    
    # Transform: normalize to [0, 1] then subtract ref_reward's normalized position
    x_transformed = ranks / (len(ranks) - 1)
    x_transformed = x_transformed - ref_rank / (len(ranks) - 1)
    
    return x_transformed.reshape(shape)


def _z_score_transform(x: jnp.ndarray) -> jnp.ndarray:
    mean = jnp.mean(x)
    std = jnp.std(x)+1e-6
    return (x - mean) / std


def _sample_bernoulli_parameter(key: Any, params: Any, sampling_dtype: Any, batch_size: Tuple = ()) -> Any:
    """Sample parameters from Bernoulli distribution. """

    num_vars = len(jax.tree_util.tree_leaves(params))
    treedef = jax.tree_util.tree_structure(params)

    all_keys = jax.random.split(key, num=num_vars)
    noise = jax.tree_util.tree_map(
        lambda p, k: jax.random.uniform(k, (*batch_size, *p.shape), sampling_dtype) < p,
        params, jax.tree_util.tree_unflatten(treedef, all_keys))

    return noise


def _deterministic_bernoulli_parameter(params: Any, batch_size: Tuple = ()) -> Any:
    """Deterministic evaluation, using p > 0.5 as True, p <= 0.5 as False"""

    return jax.tree_util.tree_map(lambda p: jnp.broadcast_to(p > 0.5, (*batch_size, *p.shape)), params)


def _bitset_param_to_prob_param(params: Any, init_p: float = 0.5, p_dtype: jnp.dtype = jnp.float32, bitset_dtype: jnp.dtype = jnp.uint32) -> Any:
    """Convert network parameters with bitset_params to probability parameters for optimization.
    
    This function handles the case where some parameters are stored as bitsets (for memory efficiency)
    and need to be converted to full probability arrays for gradient-based optimization.
    
    For bitset_params:
    - Input: {'params': {...}, 'bitset_params': {...}}
    - Output: FrozenDict({'params': {...}, 'bitset_params': {...}}) with expanded bitset dimensions
    
    For regular params:
    - Input: {'params': {...}}  (network_variables from Flax)
    - Output: {...} (extract and convert params directly)
    
    Args:
        params: Network parameters dict, may contain 'params' and 'bitset_params' keys
        init_p: Initial probability value (typically 0.5 for uniform distribution)
        p_dtype: Data type for probability parameters
        bitset_dtype: Data type for bitset (e.g., uint32 packs 32 bools)
    
    Returns:
        Probability parameters suitable for optimization
    """
    # Check if params has bitset_params (for ConnSNN_Optim)
    if isinstance(params, dict) and 'bitset_params' in params:
        from flax.core.frozen_dict import FrozenDict
        bitset_len = jnp.iinfo(bitset_dtype).bits
        bitset_shapes = jax.tree.map(lambda x: (*x.shape[:-1], x.shape[-1] * bitset_len), params['bitset_params'])
        probs_params = FrozenDict({
            'params': jax.tree.map(lambda x: jnp.full_like(x, init_p, p_dtype), params['params']),
            'bitset_params': jax.tree.map(lambda x, s: jnp.full(s, init_p, p_dtype), params['bitset_params'], bitset_shapes),
        })
        return probs_params
    else:
        # Regular params without bitset (for ConnSNN)
        # If params is the full network_variables dict with 'params' key, extract it
        if isinstance(params, dict) and 'params' in params and 'fixed_weights' in params:
            # This is network_variables from Flax init
            actual_params = params['params']
        else:
            # This is already the params dict
            actual_params = params
        return jax.tree.map(lambda x: jnp.full_like(x, init_p, p_dtype), actual_params)


# Evaluate the population for a single step
def _evaluate_step(pop: PopulationState, runner: RunnerState, conf: ESConfig, env_cls: Any = None, env_reset_pool: Any = None) -> PopulationState:
    # step env
    # NOTE: vmapping apply for multiple set of parameters (broadcast fixed weights)
    # Check if network uses bitset optimization
    # Use try/except to handle dict, FrozenDict, and other dict-like structures
    # Also check for nested structure {"params": {"params": ..., "bitset_params": ...}}
    has_bitset_params = False
    network_params_to_use = pop.network_params
    try:
        if 'bitset_params' in pop.network_params:
            has_bitset_params = True
            network_params_to_use = pop.network_params
        elif 'params' in pop.network_params:
            # Check nested structure
            inner = pop.network_params['params']
            if hasattr(inner, '__contains__') and 'bitset_params' in inner:
                has_bitset_params = True
                network_params_to_use = inner
    except (TypeError, AttributeError, KeyError):
        pass
    
    if has_bitset_params:
        # ConnSNN_Optim with bitset optimization
        vmapped_apply = jax.vmap(conf.network_cls.apply, ({"params": 0, "fixed_weights": None, 'bitset_params': 0}, 0, 0))
        
        # Convert boolean bitset_params to bitset format for efficient computation
        bitset_param = jax.tree.map(lambda x: bitset_from_bool(x, conf.bitset_dtype), network_params_to_use['bitset_params'])
        
        obs_norm = running_statistics.normalize(pop.env_states.obs, runner.normalizer_state)
        new_network_states, act = vmapped_apply(
            {"params": network_params_to_use['params'], "fixed_weights": runner.fixed_weights, 'bitset_params': bitset_param}, 
            pop.network_states, obs_norm
        )
    else:
        # Regular ConnSNN without bitset
        vmapped_apply = jax.vmap(conf.network_cls.apply, ({"params": 0, "fixed_weights": None}, 0, 0))
        
        obs_norm = running_statistics.normalize(pop.env_states.obs, runner.normalizer_state)
        # Check if network_params already has "params" key (could be dict, FrozenDict, etc.)
        # Also check for nested {"params": {"params": ...}} structure
        try:
            # Try to access 'params' key
            if 'params' in pop.network_params:
                inner_params = pop.network_params['params']
                # Check if nested (has {"params": {"params": ...}})
                try:
                    if 'params' in inner_params:
                        # Nested structure, unwrap one level
                        variables = {"params": inner_params['params'], "fixed_weights": runner.fixed_weights}
                    else:
                        # network_params is already in {"params": ...} format
                        variables = {"params": inner_params, "fixed_weights": runner.fixed_weights}
                except (TypeError, KeyError, AttributeError):
                    # inner_params is not a dict-like structure, use it directly
                    variables = {"params": inner_params, "fixed_weights": runner.fixed_weights}
            else:
                # network_params is direct params, need to wrap
                variables = {"params": pop.network_params, "fixed_weights": runner.fixed_weights}
        except (TypeError, KeyError, AttributeError):
            # Fallback: network_params is direct params, need to wrap
            variables = {"params": pop.network_params, "fixed_weights": runner.fixed_weights}
        new_network_states, act = vmapped_apply(variables, pop.network_states, obs_norm)
    
    assert act.dtype == conf.network_dtype   # Sanity check, avoid silent promotion

    act = jnp.clip(act, -1, 1)  # brax do not clip actions internally.

    # NOTE: Cast type and avoid NaNs, set them to 0
    if act.dtype != conf.action_dtype:
        act = jnp.where(jnp.isnan(act), 0, act).astype(conf.action_dtype)

    # Use provided env_cls or fall back to conf.env_cls
    actual_env_cls = env_cls if env_cls is not None else conf.env_cls
    new_env_states = actual_env_cls.step(pop.env_states, act)

    # calculate episodic rewards
    new_fitness_totrew = pop.fitness_totrew + new_env_states.reward

    new_fitness_sum    = jnp.where(new_env_states.done, pop.fitness_sum + new_fitness_totrew, pop.fitness_sum)
    new_fitness_n      = jnp.where(new_env_states.done, pop.fitness_n   + 1,                  pop.fitness_n)
    # clear tot rew
    new_fitness_totrew = jnp.where(new_env_states.done, 0, new_fitness_totrew)

    # reset done envs
    # Reference: brax / envs / wrapper.py
    def _where_done(x, y):
        done = new_env_states.done
        done = done.reshape([-1] + [1] * (len(x.shape) - 1))
        return jnp.where(done, x, y)

    # Use provided env_reset_pool or fall back to runner.env_reset_pool
    actual_reset_pool = env_reset_pool if env_reset_pool is not None else runner.env_reset_pool
    new_env_states = jax.tree.map(_where_done, actual_reset_pool, new_env_states)

    # Increment environment steps counter
    # Each call to _evaluate_step represents one environment step for the entire population
    new_env_steps = pop.env_steps + 1

    return pop.replace(
        # Network
        network_states=new_network_states,
        # Env
        env_states=new_env_states,
        # Fitness
        fitness_totrew=new_fitness_totrew,
        fitness_sum=new_fitness_sum,
        fitness_n=new_fitness_n,
        # Environment steps
        env_steps=new_env_steps
    )


@partial(jax.jit, static_argnums=(2,))
def _runner_init(key: Any, network_init_key: Any, conf: ESConfig) -> RunnerState:
    # split run keys for initializing env
    key, env_init_key, eval_env_init_key = jax.random.split(key, 3)

    # init train env (size = pop_size - eval_size, used for training samples)
    train_size = conf.pop_size - conf.eval_size
    env_reset_pool = conf.env_cls.reset(jax.random.split(env_init_key, train_size))
    
    # init eval env (use eval_env_cls if available, otherwise use env_cls)
    eval_env_cls = conf.eval_env_cls if conf.eval_env_cls is not None else conf.env_cls
    eval_env_reset_pool = eval_env_cls.reset(jax.random.split(eval_env_init_key, conf.eval_size))

    # init network params + opt state
    # For ConnSNN_Optim, we need to initialize with a single example (not batch)
    # to get the correct parameter structure
    init_carry = conf.network_cls.initial_carry(jax.random.PRNGKey(0), 1)
    init_carry = jax.tree.map(lambda x: x[0], init_carry)
    init_obs = env_reset_pool.obs[0]
    
    # Check if network supports bitset_params (ConnSNN_Optim)
    try:
        network_variables = jax.jit(conf.network_cls.init, donate_argnums=(1,))(
            {"params": network_init_key, "fixed_weights": network_init_key, "bitset_params": network_init_key},
            init_carry,
            init_obs
        )
        # Network supports bitset optimization
        network_fixed_weights = network_variables["fixed_weights"]
        # Convert bitset params to probability params for optimization
        network_params = _bitset_param_to_prob_param(network_variables, 0.5, conf.p_dtype, conf.bitset_dtype)
    except Exception:
        # Regular network without bitset optimization
        # Use train_size for initialization since that's what we'll be using
        network_variables = jax.jit(conf.network_cls.init, donate_argnums=(1,))(
            {"params": network_init_key, "fixed_weights": network_init_key},
            conf.network_cls.initial_carry(jax.random.PRNGKey(0), train_size),
            env_reset_pool.obs
        )
        # Extract params and fixed_weights from network_variables
        network_fixed_weights = network_variables["fixed_weights"]
        # network_variables["params"] contains the actual parameters
        # We convert them to probability parameters (p=0.5)
        network_params = jax.tree.map(lambda x: jnp.full_like(x, 0.5, conf.p_dtype), network_variables["params"])
    
    optim_state = conf.optim_cls.init(network_params)

    # Initialize momentum state if using momentum
    if conf.momentum > 0:
        # Initialize momentum state to zeros (same structure as params)
        momentum_state = jax.tree.map(lambda p: jnp.zeros_like(p), network_params)
    else:
        momentum_state = None

    # runner state
    # Initialize ref_reward to -inf as a sentinel value (will be replaced with actual value on first generation if use_reward_ref=True)
    initial_ref_reward = jnp.array(-jnp.inf, dtype=jnp.float32)
    runner = RunnerState(
        key=key,
        normalizer_state=running_statistics.init_state(specs.Array((conf.env_cls.observation_size, ), jnp.float32)),
        env_reset_pool=env_reset_pool,
        eval_env_reset_pool=eval_env_reset_pool,

        params=network_params,
        fixed_weights=network_fixed_weights,
        opt_state=optim_state,
        ref_reward=initial_ref_reward,
        momentum_state=momentum_state
    )
    return runner


class ScaleDeltaWithAbs:
    """Modular function class: Scale updates based on abs_param
    
    This class encapsulates the logic of scaling gradient updates based on parameter absolute values.
    When scale_delta_with_abs=True, multiplies updates by abs_param raised to scale_factor power.
    
    Note: Methods of this class can be called inside JAX JIT functions as they only use JAX array operations.
    """
    
    def __init__(self, enabled: bool = True, scale_factor: float = 0.5):
        """
        Args:
            enabled: whether to enable scaling functionality
            scale_factor: scale factor, used to compute abs_param ** scale_factor
        """
        self.enabled = enabled
        self.scale_factor = scale_factor
    
    @staticmethod
    def compute_abs_param(params: Any) -> Any:
        """Compute absolute value of parameters (static method, can be used in JIT)
        
        Args:
            params: current parameters (p value of Bernoulli distribution, range in [eps, 1-eps])
        
        Returns:
            abs_param: absolute value of parameters, computed as min(p, 1-p)
        """
        return jax.tree.map(lambda p: jnp.where(p > 0.5, 1.0 - p, p), params)
    
    def apply(self, updates: Any, abs_param: Any) -> Any:
        """Apply scaling to updates
        
        Args:
            updates: updates computed by optimizer
            abs_param: absolute value of parameters
        
        Returns:
            scaled_updates: scaled updates
        """
        if self.enabled:
            return jax.tree.map(
                lambda u, a: u * jnp.power(a, self.scale_factor),
                updates, abs_param
            )
        else:
            return updates
    
    def __call__(self, updates: Any, params: Any) -> Tuple[Any, Any]:
        """Convenient call interface: simultaneously compute abs_param and apply scaling
        
        Args:
            updates: updates computed by optimizer
            params: current parameters
        
        Returns:
            (scaled_updates, abs_param): scaled updates and abs_param
        """
        abs_param = self.compute_abs_param(params)
        scaled_updates = self.apply(updates, abs_param)
        return scaled_updates, abs_param


class SphericalTrustRegionScale:
    """Modular function class: Scale updates based on spherical trust region
    
    This class encapsulates the logic of scaling gradient updates based on spherical trust region.
    When enabled, multiplies updates by (rho*(1-rho))**scale_factor.
    The difference from ScaleDeltaWithAbs is:
    - ScaleDeltaWithAbs: uses min(rho, 1-rho)**scale_factor
    - SphericalTrustRegionScale: uses (rho*(1-rho))**scale_factor
    
    Note: Methods of this class can be called inside JAX JIT functions as they only use JAX array operations.
    """
    
    def __init__(self, enabled: bool = True, scale_factor: float = 0.5):
        """
        Args:
            enabled: whether to enable scaling functionality
            scale_factor: scale factor, used to compute (rho*(1-rho)) ** scale_factor
        """
        self.enabled = enabled
        self.scale_factor = scale_factor
    
    @staticmethod
    def compute_spherical_param(params: Any) -> Any:
        """Compute spherical trust region parameter (static method, can be used in JIT)
        
        Args:
            params: current parameters (p value of Bernoulli distribution, range in [eps, 1-eps])
        
        Returns:
            spherical_param: spherical trust region parameter, computed as p * (1-p)
        """
        return jax.tree.map(lambda p: p * (1.0 - p), params)
    
    def apply(self, updates: Any, spherical_param: Any) -> Any:
        """Apply scaling to updates
        
        Args:
            updates: updates computed by optimizer
            spherical_param: spherical trust region parameter (rho*(1-rho))
        
        Returns:
            scaled_updates: scaled updates
        """
        if self.enabled:
            return jax.tree.map(
                lambda u, s: u * jnp.power(s, self.scale_factor),
                updates, spherical_param
            )
        else:
            return updates
    
    def __call__(self, updates: Any, params: Any) -> Tuple[Any, Any]:
        """Convenient call interface: simultaneously compute spherical_param and apply scaling
        
        Args:
            updates: updates computed by optimizer
            params: current parameters
        
        Returns:
            (scaled_updates, spherical_param): scaled updates and spherical_param
        """
        spherical_param = self.compute_spherical_param(params)
        scaled_updates = self.apply(updates, spherical_param)
        return scaled_updates, spherical_param


class MetaplasticTrustRegionRSRP:
    """Modular function class: Metaplastic Trust-Region RSRP (MTR-RSRP) update method
    
    This class implements metaplastic trust region update method.Applies different
    gain factors based on gradient direction: positive gradients use (1-ρ)^α, negative gradients use ρ^α, forming asymmetric trust region effect.
    
    Note: Methods of this class can be called inside JAX JIT functions as they only use JAX array operations.
    """
    
    def __init__(self, enabled: bool = True, alpha: float = 0.3):
        """
        Args:
            enabled: whether to enable MTR-RSRP update
            alpha: metaplasticity exponent, controls gain factor strength
        """
        self.enabled = enabled
        self.alpha = alpha
    
    def compute_gain_factors(self, params: Any) -> Tuple[Any, Any]:
        """Compute metaplastic gain factors (static method, can be used in JIT)
        
        Args:
            params: current parameters (p value of Bernoulli distribution, range in [eps, 1-eps])
        
        Returns:
            (f_pot, f_dep): positive gradient gain factors and negative gradient gain factors
                f_pot = (1-ρ)^α, f_dep = ρ^α
        """
        f_pot = jax.tree.map(lambda p: jnp.power(1.0 - p, self.alpha), params)
        f_dep = jax.tree.map(lambda p: jnp.power(p, self.alpha), params)
        return f_pot, f_dep
    
    def apply(self, updates: Any, params: Any) -> Any:
        """Apply MTR-RSRP update method
        
        Args:
            updates: updates computed by optimizer（gradient direction）
            params: current parameters
        
        Returns:
            metaplastic_updates: updates after applying metaplastic scaling
        """
        if not self.enabled:
            return updates
        
        # Decompose gradient direction: positive gradient (enhancement) and negative gradient (suppression)
        updates_pos = jax.tree.map(lambda u: jnp.maximum(u, 0.0), updates)  # g^+
        updates_neg = jax.tree.map(lambda u: jnp.minimum(u, 0.0), updates)  # g^-
        
        # Compute gain factors
        f_pot, f_dep = self.compute_gain_factors(params)
        
        # Apply metaplastic update: Δρ = f_pot * g^+ + f_dep * g^-
        # Note: learning rate is already applied in optimizer, do not apply again here
        metaplastic_updates = jax.tree.map(
            lambda u_pos, u_neg, f_p, f_d: f_p * u_pos + f_d * u_neg,
            updates_pos, updates_neg, f_pot, f_dep
        )
        
        return metaplastic_updates
    
    def __call__(self, updates: Any, params: Any) -> Any:
        """Convenient call interface: Apply MTR-RSRP update
        
        Args:
            updates: updates computed by optimizer
            params: current parameters
        
        Returns:
            metaplastic_updates: updates after applying metaplastic scaling
        """
        return self.apply(updates, params)


class BoundaryAwareRSRP:
    """Modular function class: Boundary-aware Metaplastic RSRP (BA-RSRP) update method
    
    This class implements boundary-aware metaplastic update method.Based on the relationship between gradient direction and parameter position,
    divides gradient into "same-direction" and "opposite-direction" parts,
    applies different gain factors: same-direction uses s(ρ)^α, opposite-direction uses s(ρ)^β.
    
    Note: Methods of this class can be called inside JAX JIT functions as they only use JAX array operations.
    """
    
    def __init__(self, enabled: bool = True, alpha: float = 0.5, beta: float = 0.7):
        """
        Args:
            enabled: whether to enable BA-RSRP update
            alpha: exponent for same-direction updates, controls gain factor strength for same-direction updates
            beta: exponent for opposite-direction updates, controls gain factor strength for opposite-direction updates
        """
        self.enabled = enabled
        self.alpha = alpha
        self.beta = beta
    
    @staticmethod
    def compute_boundary_distance(params: Any) -> Any:
        """Compute distance from parameters to nearest boundary (static method, can be used in JIT)
        
        Args:
            params: current parameters (p value of Bernoulli distribution, range in [eps, 1-eps])
        
        Returns:
            s: boundary distance, computed as min(ρ, 1-ρ)
        """
        return jax.tree.map(lambda p: jnp.minimum(p, 1.0 - p), params)
    
    def compute_direction_masks(self, updates: Any, params: Any) -> Tuple[Any, Any]:
        """Compute same-direction and opposite-direction masks
        
        Args:
            updates: updates computed by optimizer（gradient direction）
            params: current parameters
        
        Returns:
            (same_mask, opp_mask): same-direction mask and opposite-direction mask
                same_mask = I[ρ < 0.5 ∧ g < 0] + I[ρ ≥ 0.5 ∧ g > 0]
                opp_mask = 1 - same_mask
        """
        # Same-direction mask: parameter < 0.5 and gradient < 0, or parameter ≥ 0.5 and gradient > 0
        same_mask = jax.tree.map(
            lambda u, p: ((p < 0.5) & (u < 0.0)) | ((p >= 0.5) & (u > 0.0)),
            updates, params
        )
        # Opposite-direction mask: other cases
        opp_mask = jax.tree.map(lambda m: ~m, same_mask)
        return same_mask, opp_mask
    
    def apply(self, updates: Any, params: Any) -> Any:
        """Apply BA-RSRP update method
        
        Args:
            updates: updates computed by optimizer（gradient direction）
            params: current parameters
        
        Returns:
            boundary_aware_updates: updates after applying boundary-aware metaplastic scaling
        """
        if not self.enabled:
            return updates
        
        # 1. Compute boundary distance
        s = self.compute_boundary_distance(params)
        
        # 2. Compute same-direction and opposite-direction masks
        same_mask, opp_mask = self.compute_direction_masks(updates, params)
        
        # 3. Decompose gradient: same-direction and opposite-direction
        updates_same = jax.tree.map(lambda u, m: jnp.where(m, u, 0.0), updates, same_mask)
        updates_opp = jax.tree.map(lambda u, m: jnp.where(m, u, 0.0), updates, opp_mask)
        
        # 4. Compute gain factors
        f_same = jax.tree.map(lambda s_val: jnp.power(s_val, self.alpha), s)
        f_opp = jax.tree.map(lambda s_val: jnp.power(s_val, self.beta), s)
        
        # 5. Apply boundary-aware update: Δρ = f_same * g_same + f_opp * g_opp
        # Note: learning rate is already applied in optimizer, do not apply again here
        boundary_aware_updates = jax.tree.map(
            lambda u_same, u_opp, f_s, f_o: f_s * u_same + f_o * u_opp,
            updates_same, updates_opp, f_same, f_opp
        )
        
        return boundary_aware_updates
    
    def __call__(self, updates: Any, params: Any) -> Any:
        """Convenient call interface: Apply BA-RSRP update
        
        Args:
            updates: updates computed by optimizer
            params: current parameters
        
        Returns:
            boundary_aware_updates: updates after applying boundary-aware metaplastic scaling
        """
        return self.apply(updates, params)


class DynamicSampling:
    """Modular function class: Dynamic sampling logic
    
    This class encapsulates dynamic sampling logic.When all training samples have reward below ref_reward,
    skips parameter update and keeps current parameters unchanged.
    
    Note: Methods of this class can be called inside JAX JIT functions as they only use JAX array operations.
    """
    
    def __init__(self, enabled: bool = True, use_reward_ref: bool = True):
        """
        Args:
            enabled: whether to enable dynamic sampling
            use_reward_ref: whether to use reference reward (must be True to enable dynamic sampling)
        """
        self.enabled = enabled
        self.use_reward_ref = use_reward_ref
    
    def should_skip_update(
        self,
        fitness: jnp.ndarray,
        ref_reward: jnp.ndarray
    ) -> jnp.ndarray:
        """Determine whether to skip update
        
        Args:
            fitness: fitness values of training samples
            ref_reward: reference reward value
        
        Returns:
            skip_flag: boolean array, True means should skip update
        """
        if not self.enabled or not self.use_reward_ref:
            return jnp.array(False)
        
        # Check if ref_reward is valid (not -inf)
        ref_reward_is_valid = ~jnp.isinf(ref_reward)
        
        # If ref_reward is valid and all fitness are below ref_reward, skip update
        all_rewards_below_ref = jnp.where(
            ref_reward_is_valid,
            jnp.all(fitness < ref_reward),
            jnp.array(False)
        )
        
        return all_rewards_below_ref
    
    def apply(
        self,
        runner: RunnerState,
        new_params: Any,
        new_opt_state: Any,
        skip_flag: jnp.ndarray,
        current_ref_reward: jnp.ndarray
    ) -> RunnerState:
        """Apply dynamic sampling logic to runner state
        
        Args:
            runner: current runner state
            new_params: newly computed parameters
            new_opt_state: newly computed optimizer state
            skip_flag: flag indicating whether to skip update
            current_ref_reward: current reference reward
        
        Returns:
            updated_runner: updated runner state
        """
        if self.enabled:
            # If skip_flag is True, keep old parameters and optimizer state
            runner_updated_params = jax.tree.map(
                lambda old, new: jnp.where(skip_flag, old, new),
                runner.params, new_params
            )
            runner_updated_opt_state = jax.tree.map(
                lambda old, new: jnp.where(skip_flag, old, new),
                runner.opt_state, new_opt_state
            )
            return runner.replace(
                params=runner_updated_params,
                opt_state=runner_updated_opt_state,
                ref_reward=current_ref_reward
            )
        else:
            return runner.replace(
                params=new_params,
                opt_state=new_opt_state,
                ref_reward=current_ref_reward
            )
    
    def __call__(
        self,
        runner: RunnerState,
        fitness: jnp.ndarray,
        new_params: Any,
        new_opt_state: Any,
        current_ref_reward: jnp.ndarray
    ) -> Tuple[RunnerState, jnp.ndarray]:
        """Convenient call interface: determine and apply dynamic sampling in one step
        
        Args:
            runner: current runner state
            fitness: fitness values of training samples
            new_params: newly computed parameters
            new_opt_state: newly computed optimizer state
            current_ref_reward: current reference reward
        
        Returns:
            (updated_runner, skip_flag): updated runner state and skip flag
        """
        skip_flag = self.should_skip_update(fitness, runner.ref_reward)
        updated_runner = self.apply(
            runner, new_params, new_opt_state, skip_flag, current_ref_reward
        )
        return updated_runner, skip_flag


@partial(jax.jit, donate_argnums=(0,), static_argnums=(1,))
def _runnner_run(
    runner: RunnerState,
    conf: ESConfig
) -> Tuple[RunnerState, Dict, Dict]:
    metrics = {}

    # split keys for this run
    new_key, run_key, carry_key = jax.random.split(runner.key, 3)
    runner = runner.replace(key=new_key)

    # Generate params with bernoulli distribution
    train_params = _sample_bernoulli_parameter(run_key, runner.params, conf.network_dtype, (conf.pop_size - conf.eval_size, ))
    eval_params  = _deterministic_bernoulli_parameter(runner.params, (conf.eval_size, ))

    # Split keys for train and eval
    train_carry_key, eval_carry_key = jax.random.split(carry_key, 2)

    # Determine eval environment class
    eval_env_cls = conf.eval_env_cls if conf.eval_env_cls is not None else conf.env_cls

    # Initialize training population
    train_pop = PopulationState(
        # Network
        network_params=train_params,
        network_states=conf.network_cls.initial_carry(train_carry_key, conf.pop_size - conf.eval_size),
        # Env
        env_states=runner.env_reset_pool,
        # Fitness
        fitness_totrew=jnp.zeros(conf.pop_size - conf.eval_size),
        fitness_sum=jnp.zeros(conf.pop_size - conf.eval_size),
        fitness_n=jnp.zeros(conf.pop_size - conf.eval_size, dtype=jnp.int32),
        # Environment steps (initialized to 0 for this generation)
        env_steps=jnp.array(0, dtype=jnp.int32)
    )

    # Initialize evaluation population
    eval_pop = PopulationState(
        # Network
        network_params=eval_params,
        network_states=conf.network_cls.initial_carry(eval_carry_key, conf.eval_size),
        # Env
        env_states=runner.eval_env_reset_pool,
        # Fitness
        fitness_totrew=jnp.zeros(conf.eval_size),
        fitness_sum=jnp.zeros(conf.eval_size),
        fitness_n=jnp.zeros(conf.eval_size, dtype=jnp.int32),
        # Environment steps (initialized to 0 for this generation)
        env_steps=jnp.array(0, dtype=jnp.int32)
    )

    # (PNN) Run some steps to warm up states (only for training)
    warmup_env_steps = jnp.array(0, dtype=jnp.int32)
    if conf.warmup_steps > 0:
        train_pop, _ = jax.lax.scan(
            lambda p, x: (_evaluate_step(p, runner, conf, conf.env_cls, runner.env_reset_pool), None), 
            train_pop, None, length=conf.warmup_steps
        )

        warmup_fitness = train_pop.fitness_sum / train_pop.fitness_n
        # Save warmup env_steps before resetting
        warmup_env_steps = train_pop.env_steps
        metrics.update({
            "warmup_fitness": finitemean(warmup_fitness)
        })

        # (PNN) Update normalizer using warmup data
        runner = runner.replace(normalizer_state=running_statistics.update(runner.normalizer_state, train_pop.env_states.obs))
        # (PNN) Reset envs + Clear fitness
        train_pop = train_pop.replace(
            # Env
            env_states=runner.env_reset_pool,
            # Fitness
            fitness_totrew=jnp.zeros(conf.pop_size - conf.eval_size),
            fitness_sum=jnp.zeros(conf.pop_size - conf.eval_size),
            fitness_n=jnp.zeros(conf.pop_size - conf.eval_size, dtype=jnp.int32),
            # Reset environment steps counter for actual evaluation
            env_steps=jnp.array(0, dtype=jnp.int32)
        )

    # Evaluate training population
    def _eval_stop_cond_train(p: PopulationState) -> jnp.ndarray:
        # Stop when finished
        return ~jnp.all(p.fitness_n >= 1)

    train_pop = jax.lax.while_loop(
        _eval_stop_cond_train, 
        lambda p: _evaluate_step(p, runner, conf, conf.env_cls, runner.env_reset_pool), 
        train_pop
    )

    # Evaluate evaluation population
    def _eval_stop_cond_eval(p: PopulationState) -> jnp.ndarray:
        # Stop when finished
        return ~jnp.all(p.fitness_n >= 1)

    eval_pop = jax.lax.while_loop(
        _eval_stop_cond_eval, 
        lambda p: _evaluate_step(p, runner, conf, eval_env_cls, runner.eval_env_reset_pool), 
        eval_pop
    )

    # Update normalizer using terminal states (only from training)
    # FIXME: May be biased towards states near episode terminal
    if conf.warmup_steps <= 0:
        runner = runner.replace(normalizer_state=running_statistics.update(runner.normalizer_state, train_pop.env_states.obs))

    # Calculate population metrics (from training population)
    if hasattr(conf.network_cls, "carry_metrics"):
        metrics.update(conf.network_cls.carry_metrics(train_pop.network_states))

    # Calculate fitness
    fitness = train_pop.fitness_sum / train_pop.fitness_n
    eval_fitness = eval_pop.fitness_sum / eval_pop.fitness_n

    # Calculate environment steps for this generation
    # Each environment step processes the entire population (pop_size environments)
    # Use training steps for total count (eval steps are separate)
    # Include warmup steps if warmup was performed
    generation_env_steps = train_pop.env_steps + warmup_env_steps
    # Update total environment steps (only training steps, not eval steps)
    new_total_env_steps = runner.total_env_steps + generation_env_steps

    # Update ref_reward if use_reward_ref is enabled
    # Always use best eval fitness as ref_reward
    if conf.use_reward_ref:
        current_ref_reward = jnp.max(eval_fitness)
    else:
        current_ref_reward = runner.ref_reward

    # ========== Gradient Computation (Natural Evolution Strategy - NES) ==========
    #
    # grad computation pipeline:
    # 1. First transform fitness values into weights:
    #    - fitness: fitness values of each individual (reward), shape (pop_size - eval_size,)
    #    - weight: transformed weights used for gradient estimation
    #       * centered_rank: transform fitness to centered rank (approximately in range [-0.5, 0.5])
    #       * z_score: transform fitness to z-score (standardized, mean 0 and std 1)
    #
    # 2. Use NES to estimate the gradient:
    #    - p: current parameters (runner.params), Bernoulli distribution parameters, range [eps, 1-eps]
    #    - theta: sampled network parameters (train_pop.network_params), shape (pop_size - eval_size, ...)
    #      each theta[i] is a binary parameter vector sampled from Bernoulli(p)
    #
    # 3. Gradient formula (Natural Gradient for Bernoulli):
    #    grad = -mean(weight * (theta - p), axis=0)
    #
    #    Explanation:
    #    - (theta - p): sampling deviation, indicating the offset of samples relative to p
    #    - weight * (theta - p): sampling deviation weighted by fitness
    #    - mean(..., axis=0): average over all individuals to obtain the expected gradient
    #    - negative sign: we perform gradient ascent (maximize fitness), hence the minus sign
    #
    # 4. This formula is the natural gradient estimator of Natural Evolution Strategy (NES)
    #    for Bernoulli distributions; see: https://arxiv.org/pdf/1703.03864.pdf
    #
    # Reconstruct noise using network parameters (choose transform)
    if conf.transform_type == "centered_rank":
        # Choose transform based on use_reward_ref and ref_reward validity
        if conf.use_reward_ref:
            # Need to handle traced array ref_reward_is_valid
            ref_reward_is_valid = ~jnp.isinf(runner.ref_reward)
            weight_with_ref = _centered_rank_transform_with_ref(fitness, runner.ref_reward)
            weight_normal = _centered_rank_transform(fitness)
            weight = jnp.where(ref_reward_is_valid, weight_with_ref, weight_normal)
        else:
            weight = _centered_rank_transform(fitness)
    elif conf.transform_type == "z_score":
        weight = _z_score_transform(fitness)
    else:
        raise ValueError(f"Invalid transform type: {conf.transform_type}")
    
    def _nes_grad(p, theta):
        """
        Natural Evolution Strategy (NES) gradient estimator.
        
        Args:
            p: current parameters (Bernoulli distribution parameters), shape (...)
            theta: sampled network parameters, shape (pop_size, ...)
        
        Returns:
            grad: estimated gradient, same shape as p
        """
        # Reshape weight to be broadcast-compatible with theta:
        # weight shape: (pop_size,)
        # theta shape: (pop_size, ...)
        # after reshape: (pop_size, 1, 1, ...) for broadcasting
        w = weight.reshape((-1,) + (1,) * (theta.ndim - 1)).astype(p.dtype)
        
        # NES gradient formula: grad = -mean(weight * (theta - p), axis=0)
        # This is the natural gradient for a Bernoulli distribution.
        return -jnp.mean(w * (theta - p), axis=0)
    
    # Compute gradients for each parameter:
    # runner.params: current parameters (p)
    # train_pop.network_params: sampled parameters (theta)
    grads = jax.tree.map(lambda p, theta: _nes_grad(p, theta), runner.params, train_pop.network_params)

    # ========== TRPO-style Fixed KL / Fixed-NS Update ==========
    if conf.use_trpo_update:
        # Natural direction: F^{-1} g for Bernoulli probs
        # F^{-1} = diag(p * (1-p)) for Bernoulli distribution
        nat_dir = jax.tree.map(lambda p, g: p * (1.0 - p) * g, runner.params, grads)
        
        # Compute g^T F^{-1} g = sum_i p_i(1-p_i) g_i^2  (tree-summed)
        def tree_sum_squares_weighted(params, grads):
            leaves = jax.tree_util.tree_leaves(
                jax.tree.map(lambda p, g: (p * (1.0 - p)) * (g * g), params, grads)
            )
            return jnp.sum(jnp.stack([jnp.sum(x) for x in leaves]))
        
        quad_form = tree_sum_squares_weighted(runner.params, grads)
        
        # Global scaling to satisfy approx KL <= kl_eps
        # alpha = sqrt(2*kl_eps / (g^T F^{-1} g + tau))
        tau = conf.trpo_damping  # numerical stability
        kl_eps = conf.trpo_kl_eps  # fixed trust-region radius (epsilon)
        alpha = jnp.sqrt((2.0 * kl_eps) / (quad_form + tau))
        
        # Clip alpha to maximum if specified (prevents overly large steps when quad_form is small)
        if conf.trpo_alpha_max is not None:
            alpha = jnp.clip(alpha, 0.0, conf.trpo_alpha_max)
        
        # Parameter update (replaces optimizer step for TRPO)
        # Note: grads is already in gradient ascent direction (negative sign in _nes_grad),
        # nat_dir = F^{-1} * grads is also in ascent direction,
        # However, since we're doing gradient ascent (maximizing fitness), we need to add nat_dir directly
        # But if performance is decreasing, the sign might be wrong - let's check by adding negative sign
        trpo_updates = jax.tree.map(lambda p, d: -alpha * d, runner.params, nat_dir)
        
        # Apply momentum if enabled (same as original method)
        if conf.momentum > 0:
            if runner.momentum_state is None:
                momentum_state = jax.tree.map(lambda p: jnp.zeros_like(p), runner.params)
            else:
                momentum_state = runner.momentum_state
            
            new_momentum_state = jax.tree.map(
                lambda prev_v, delta: conf.momentum * prev_v + delta,
                momentum_state,
                trpo_updates
            )
            updates = new_momentum_state
        else:
            updates = trpo_updates
            new_momentum_state = runner.momentum_state
        
        # Update opt_state for compatibility (not used in TRPO update)
        _, new_opt_state = conf.optim_cls.update(grads, runner.opt_state, runner.params)
        
        # Compute abs_param for logging
        abs_param = ScaleDeltaWithAbs.compute_abs_param(runner.params)
        
        # Apply updates
        scaled_updates = updates
        new_params_preclip = optax.apply_updates(runner.params, scaled_updates)
        
        # Add TRPO-specific metrics
        # Compute effective KL (actual KL divergence from the update)
        # KL ≈ (1/2) * alpha^2 * quad_form (approximate)
        effective_kl = 0.5 * alpha * alpha * quad_form
        metrics.update({
            "trpo_alpha": alpha,
            "trpo_quad_form": quad_form,
            "trpo_kl_eps": kl_eps,
            "trpo_effective_kl": effective_kl,  # Actual KL from update (may be less than kl_eps if alpha was clipped)
            "trpo_alpha_clipped": jnp.where(
                conf.trpo_alpha_max is not None and alpha >= conf.trpo_alpha_max,
                1.0, 0.0
            ) if conf.trpo_alpha_max is not None else 0.0
        })
    
    # ========== Original Custom Update Formula: delta = -sign(grad) * sqrt(2*lr*(1-p)*p) ==========
    else:
        # Compute sign(grad)
        sign_grad = jax.tree.map(lambda g: jnp.sign(g), grads)
        
        # Compute sqrt(2*lr*(1-p)*p), where p is runner.params
        # Note: p*(1-p) is the variance term of Bernoulli distribution
        sqrt_factor = jax.tree.map(
            lambda p: jnp.sqrt(2.0 * conf.lr * (1.0 - p) * p),
            runner.params
        )
        
        # Compute delta = -sign(grad) * sqrt(2*lr*(1-p)*p)
        # Note: negative sign is for gradient ascent (maximizing fitness)
        custom_updates = jax.tree.map(
            lambda s, f: -s * f,
            sign_grad, sqrt_factor
        )
        
        # ========== Apply Momentum (if enabled) ==========
        # Momentum Formula: v_t = momentum * v_{t-1} + custom_update
        # where:
        #   - v_t: momentum update at current time step
        #   - v_{t-1}: momentum state at previous time step
        #   - custom_update: update computed by custom update formula
        #   - momentum: momentum coefficient (typically around 0.9)
        #
        # Benefits of using momentum:
        # 1. Smooth updates: reduce noise impact, make training more stable
        # 2. Accelerate convergence: can accumulate velocity when gradient direction is consistent
        # 3. Help escape local optima: momentum can provide inertia when gradient direction changes
        
        if conf.momentum > 0:
            # Ensure momentum_state is initialized
            if runner.momentum_state is None:
                # If momentum_state is not initialized, initialize to zeros (should not happen, but for robustness)
                momentum_state = jax.tree.map(lambda p: jnp.zeros_like(p), runner.params)
            else:
                momentum_state = runner.momentum_state
            
            # Manually implement momentum: v_t = momentum * v_{t-1} + custom_update
            new_momentum_state = jax.tree.map(
                lambda prev_v, delta: conf.momentum * prev_v + delta,
                momentum_state,
                custom_updates
            )
            # Use updates after momentum
            updates = new_momentum_state
        else:
            # No momentum, directly use custom update
            updates = custom_updates
            # Keep momentum_state unchanged (if exists)
            new_momentum_state = runner.momentum_state
        
        # Update optimizer state (through optimizer, to maintain state consistency)
        # Note: Although we use custom updates, we still need to update opt_state for compatibility
        # opt_state is mainly used to maintain optax internal state (such as trace, etc.), even if we don't directly use its output
        # Important: Here we call optim_cls.update() but **do not use its return value** (ignore with _),
        #        because we use the custom_updates computed above (or updates after momentum)
        _, new_opt_state = conf.optim_cls.update(grads, runner.opt_state, runner.params)
        
        # Compute abs_param (for logging)
        abs_param = ScaleDeltaWithAbs.compute_abs_param(runner.params)
        
        # ========== Confirm using custom update formula ==========
        # Important: Here we directly use the updates computed above (from custom_updates or after momentum processing)
        # No longer use optimizer returned updates, and do not apply additional scaling based on update_method
        # Update formula: delta = -sign(grad) * sqrt(2*lr*(1-p)*p) [+ momentum]
        scaled_updates = updates
        
        # Apply updates to parameters
        new_params_preclip = optax.apply_updates(runner.params, scaled_updates)

    # Collect pre-clip delta (already includes scaling effect if enabled)
    delta_preclip = jax.tree.map(lambda new, old: new - old, new_params_preclip, runner.params)

    # Clip to Bernoulli range with exploration
    new_params = jax.tree.map(lambda p: jnp.clip(p, conf.eps, 1 - conf.eps), new_params_preclip)

    # Use modular DynamicSampling to handle dynamic sampling
    # Determine whether to skip update
    if conf.dynamic_sampling and conf.use_reward_ref:
        ref_reward_is_valid = ~jnp.isinf(runner.ref_reward)
        all_rewards_below_ref = jnp.where(
            ref_reward_is_valid,
            jnp.all(fitness < runner.ref_reward),
            jnp.array(False)
        )
    else:
        all_rewards_below_ref = jnp.array(False)
    
    # Apply dynamic sampling logic
    if conf.dynamic_sampling:
        runner_updated_params = jax.tree.map(
            lambda old, new: jnp.where(all_rewards_below_ref, old, new),
            runner.params, new_params
        )
        runner_updated_opt_state = jax.tree.map(
            lambda old, new: jnp.where(all_rewards_below_ref, old, new),
            runner.opt_state, new_opt_state
        )
        # If using momentum, momentum_state also needs to be updated
        if conf.momentum > 0 and new_momentum_state is not None:
            runner_updated_momentum_state = jax.tree.map(
                lambda old, new: jnp.where(all_rewards_below_ref, old, new),
                runner.momentum_state, new_momentum_state
            )
        else:
            runner_updated_momentum_state = new_momentum_state
        runner_updated = runner.replace(
            params=runner_updated_params,
            opt_state=runner_updated_opt_state,
            ref_reward=current_ref_reward,
            total_env_steps=new_total_env_steps,
            momentum_state=runner_updated_momentum_state
        )
    else:
        runner_updated = runner.replace(
            params=new_params,
            opt_state=new_opt_state,
            ref_reward=current_ref_reward,
            total_env_steps=new_total_env_steps,
            momentum_state=new_momentum_state
        )
    
    # aux is always computed (will be replaced in main if skip occurred)
    aux = {
        "delta_preclip": delta_preclip,
        "abs_param": abs_param,
        "params_preclip": new_params_preclip,
        "params_postclip": new_params
    }

    # Metrics
    metrics.update({
        "fitness":        jnp.mean(fitness),
        "eval_fitness":   jnp.mean(eval_fitness),
        "sparsity": mean_weight_abs(new_params),
        "generation_env_steps": jnp.asarray(generation_env_steps),
        "total_env_steps": jnp.asarray(new_total_env_steps)
    })
    
    # Add ref_reward metric if using reference reward
    if conf.use_reward_ref:
        metrics.update({
            "ref_reward": jnp.asarray(current_ref_reward)
        })
    
    # Add skip indicator if dynamic_sampling
    if conf.dynamic_sampling:
        metrics.update({
            "dynamic_sampling_skip": jnp.asarray(all_rewards_below_ref)
        })

    return runner_updated, metrics, aux


def main(conf):
    conf = OmegaConf.merge({
        # Task
        "seed": 1,
        "task": "humanoid",
        # "task": "inverted_pendulum",
        "task_conf": {
        },
        "episode_conf": {
            "max_episode_length": 1000,
            "action_repeat": 1
        },
        "eval_episode_conf": {
            "max_episode_length": 1000,  # Can be different from training
            "action_repeat": 1
        },

        # Train & Checkpointing
        "total_generations": 2000,
        "save_every": 1000,

        # Network
        "network_type": "ConnSNN_Optim",  # Default to bitset-optimized version
        "network_conf": {},

        # ES hyperparameter (see ESConfig)
        "es_conf": {}
    }, conf)
    # Naming - Select different parameter display based on update_method
    update_method = conf.es_conf.get("update_method", "scale_delta_with_abs")
    if update_method == "scale_delta_with_abs":
        method_param = f"sf{conf.es_conf.get('scale_factor', 0.5)}"
    elif update_method == "mtr_rsrp":
        method_param = f"ma{conf.es_conf.get('metaplastic_alpha', 0.3)}"
    elif update_method == "ba_rsrp":
        ba_alpha = conf.es_conf.get('ba_alpha', 0.5)
        ba_beta = conf.es_conf.get('ba_beta', 0.7)
        method_param = f"ba{ba_alpha}b{ba_beta}"
    elif update_method == "spherical_trust_region":
        method_param = f"str{conf.es_conf.get('scale_factor', 0.5)}"
    else:
        method_param = ""
    
    # Add bitset indicator to run name if using ConnSNN_Optim
    bitset_indicator = "bitset" if conf.network_type == "ConnSNN_Optim" else ""
    
    # Add custom update indicator
    custom_update_indicator = "sgd_custom"
    
    conf = OmegaConf.merge({
        "project_name": f"E-SNN-KL-{conf.task}",
        "run_name":     f"EC TRPO {bitset_indicator} {conf.seed} pop{conf.es_conf.pop_size} kl{conf.es_conf.trpo_kl_eps} lr{conf.es_conf.lr} {custom_update_indicator} {time.strftime('%H:%M %m-%d')}"
    }, conf)
    # ES Config
    es_conf = ESConfig(**conf.es_conf)

    print(OmegaConf.to_yaml(conf))
    print(es_conf)

    # create train env cls
    train_env = envs.get_environment(conf.task, **conf.task_conf)
    train_env = envs.wrappers.EpisodeWrapper(train_env, conf.episode_conf.max_episode_length, conf.episode_conf.action_repeat)
    train_env = envs.wrappers.VmapWrapper(train_env)
    
    # create eval env cls (can have different max_episode_length)
    eval_env = envs.get_environment(conf.task, **conf.task_conf)
    eval_env = envs.wrappers.EpisodeWrapper(eval_env, conf.eval_episode_conf.max_episode_length, conf.eval_episode_conf.action_repeat)
    eval_env = envs.wrappers.VmapWrapper(eval_env)
    
    # For backward compatibility, use train_env as default env
    env = train_env

    # create network cls
    network_cls = NETWORKS[conf.network_type]
    # Check if network supports bitset optimization (ConnSNN_Optim)
    if conf.network_type == "ConnSNN_Optim":
        network = network_cls(
            out_dims=env.action_size,
            neuron_dtype=es_conf.network_dtype,
            bitset_dtype=es_conf.bitset_dtype,
            **conf.network_conf
        )
    else:
        network = network_cls(
            out_dims=env.action_size,
            neuron_dtype=es_conf.network_dtype,
            **conf.network_conf
        )

    # create optim cls - use SGD with momentum instead of Adam
    # Note: Since we use the custom update formula, the optimizer is mainly used to maintain momentum state (if needed)
    optim = optax.chain(
        optax.trace(decay=es_conf.momentum, nesterov=False),  # SGD momentum
        (optax.add_decayed_weights(es_conf.weight_decay) if es_conf.weight_decay > 0 else optax.identity()),
        optax.scale(-es_conf.lr)  # Note: this learning rate will not be directly used in custom update, but kept for compatibility
    )

    # [initialize]
    # initialize cls in es conf
    es_conf = es_conf.replace(
        network_cls=network,
        optim_cls=optim,
        env_cls=train_env,
        eval_env_cls=eval_env
    )

    # Create modular processors (for documentation and possible future extensions)
    # Note: inside JIT functions, processing logic directly uses configuration parameters
    # These processor classes provide clear modular interfaces that can be used elsewhere
    if es_conf.update_method == "scale_delta_with_abs":
        scale_delta_handler = ScaleDeltaWithAbs(
            enabled=es_conf.scale_delta_with_abs,
            scale_factor=es_conf.scale_factor
        )
    elif es_conf.update_method == "mtr_rsrp":
        mtr_rsrp_handler = MetaplasticTrustRegionRSRP(
            enabled=True,
            alpha=es_conf.metaplastic_alpha
        )
    elif es_conf.update_method == "ba_rsrp":
        ba_rsrp_handler = BoundaryAwareRSRP(
            enabled=True,
            alpha=es_conf.ba_alpha,
            beta=es_conf.ba_beta
        )
    elif es_conf.update_method == "spherical_trust_region":
        spherical_handler = SphericalTrustRegionScale(
            enabled=True,
            scale_factor=es_conf.scale_factor
        )
    dynamic_sampling_handler = DynamicSampling(
        enabled=es_conf.dynamic_sampling,
        use_reward_ref=es_conf.use_reward_ref
    )

    # runner state
    key_run, key_network_init = jax.random.split(jax.random.PRNGKey(conf.seed))
    runner = _runner_init(key_run, key_network_init, es_conf)

    # save model path
    conf.save_model_path = "models/{}/{}/".format(conf.project_name, conf.run_name)

    # wandb
    # Include ESConfig primitive (str/int/float) fields in wandb config
    wandb_config = OmegaConf.to_container(conf)
    try:
        es_conf_dict = dataclasses.asdict(es_conf)
        def _is_simple_type(v):
            return isinstance(v, (str, float)) or (isinstance(v, int) and not isinstance(v, bool))
        es_conf_simple = {k: v for k, v in es_conf_dict.items() if _is_simple_type(v)}
        # Avoid top-level collisions by namespacing
        wandb_config["es_conf_runtime"] = es_conf_simple
    except Exception:
        pass

    if "log_group" in conf:
        init_wandb_safe(
            project=f"(G) E-SNN-{conf.task}",
            group=conf.log_group,
            name=str(conf.seed),
            config=wandb_config
        )
    else:
        init_wandb_safe(
            project=conf.project_name,
            name=conf.run_name,
            config=wandb_config
        )

    # run
    # Store previous metrics and aux for dynamic_sampling
    prev_metrics = None
    prev_aux = None
    
    for step in tqdm(range(1, conf.total_generations + 1)):
        runner, metrics, aux = _runnner_run(runner, es_conf)

        metrics = jax.device_get(metrics)
        aux = jax.device_get(aux)
        
        # Check if dynamic_sampling skip occurred
        skip_value = metrics.get("dynamic_sampling_skip", False)
        # Convert to Python bool (handles numpy array/scalar)
        skip_occurred = bool(np.asarray(skip_value).item()) if skip_value is not False else False
        if es_conf.dynamic_sampling and step > 1 and skip_occurred and prev_metrics is not None:
            # Use previous metrics and aux
            metrics = prev_metrics
            aux = prev_aux
        else:
            # Store current metrics and aux for potential future skip
            prev_metrics = metrics.copy()
            prev_aux = aux.copy()

        wandb.log(metrics, step=step)

        if not (step % conf.save_every):
            fn = conf.save_model_path + str(step)
            save_obj_to_file(fn, dict(
                conf=conf,
                state=dict(
                    normalizer_state=runner.normalizer_state,
                    fixed_weights=runner.fixed_weights,
                    params=runner.params
                )
            ))

            wandb.save(fn)

    return metrics


def sweep(seed: int, conf_override: OmegaConf):
    # pop_sizes = [8192]
    pop_sizes = [256,512,1024]
    lrs = [0.15]
    trpo_kl_eps_list = [0.1,0.3,1,3,10,30,100,300,1000,3000]
    """
    the real kl will be trpo_kl_eps * n_params/2, 
    because the avg rho of all params will always be extremely close to 1/2 during the training when the init rho is 1/2
    so the real kl is trpo_kl_eps /( n_params/2)
    the n_params is 193k, aproximate 200k, so the real kl is trpo_kl_eps / (200k/2) = trpo_kl_eps / 100k
    """
    real_kl_list = [trpo_kl_eps / 100000 for trpo_kl_eps in trpo_kl_eps_list]
    # scale_factors = [0.5]
    # metaplastic_alphas = [0.05,0.1,0.2]
    epss = [1e-3]
    transform_types = [ "centered_rank"]
    seeds = [i for i in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49]]  # explicitly specify seed values to iterate over
    def _objective(trial: optuna.Trial):
        seed = trial.suggest_categorical("seed", seeds)
        transform_type = trial.suggest_categorical("transform_type", transform_types)
        pop_size = trial.suggest_categorical("pop_size", pop_sizes)
        lr = trial.suggest_categorical("lr", lrs)
        trpo_kl_eps = trial.suggest_categorical("trpo_kl_eps", trpo_kl_eps_list)
        # metaplastic_alpha = trial.suggest_categorical("metaplastic_alpha", metaplastic_alphas)
        eps = trial.suggest_categorical("eps", epss)    
        # automatically set eval_size to ensure training subset size is at least 2
        eval_size = min(128, pop_size // 4)
        if eval_size >= pop_size:
            eval_size = pop_size - 2
        if pop_size - eval_size < 2:
            eval_size = pop_size - 2

        conf = OmegaConf.merge(conf_override, {
            "seed": seed,
                "es_conf": {
                "pop_size": pop_size,
                "eval_size": eval_size,
                "use_trpo_update": True,
                "trpo_kl_eps": trpo_kl_eps,
                # "update_method": "s",
                # "metaplastic_alpha": metaplastic_alpha,
                "lr": lr,
                "eps": eps,
                "transform_type": transform_type

            }
        })

        metrics = main(conf)
        return metrics["eval_fitness"]

    # Define grid search space: must include all parameters suggested in _objective
    search_space = {
        "seed": seeds,
        "trpo_kl_eps": trpo_kl_eps_list,
        "transform_type": transform_types,
        "pop_size": pop_sizes,
        "lr": lrs,
        # "metaplastic_alpha": metaplastic_alphas,
        "eps": epss
    }

    # Create Study using grid sampling
    study = optuna.create_study(
        direction="maximize",
        sampler=GridSampler(search_space)
    )

    # Run optimization, automatically iterate over all pop_size combinations
    study.optimize(_objective)


if __name__ == "__main__":
    _config = OmegaConf.from_cli()
    if hasattr(_config, "sweep"):
        sweep(_config.sweep, _config)
    else:
        main(_config)
