# Signal-Adaptive Trust Regions

**Gradient-free optimization of recurrent spiking neural networks for continuous control.**

This repository implements Signal-Adaptive Trust Regions for Evolving Connectivity (SATR-EC), along with EC, fixed-trust-region EC, and dense evolution-strategy baselines. SATR adapts the update scale to the current Bernoulli connectivity probabilities. The optimized connectivity network packs recurrent connections into bitsets.

## Setup

Use Python with JAX and a compatible Brax v1 installation. Install JAX for your hardware following the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html), then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

A tested configuration uses JAX 0.4.35 and Brax 0.1.2. The requirements are not fully pinned; other combinations may need adjustments. GPU is recommended for full experiments. Weights & Biases (W&B) records metrics; authenticate for online logging, or set `WANDB_MODE=disabled` for a local smoke test.

## Quick start

Run a small CPU smoke test (four total individuals, including one evaluation individual; two generations and eight steps per episode):

```bash
# Activate your Python environment first
WANDB_MODE=disabled JAX_PLATFORMS=cpu python ec_satr_sweep.py \
  task=hopper es_conf.pop_size=4 es_conf.eval_size=1 \
  network_conf.num_neurons=32 total_generations=2 \
  episode_conf.max_episode_length=8 eval_episode_conf.max_episode_length=8
```

This checks that training runs; it is **not** a reproduction of the paper's results. The optimized network requires `network_conf.num_neurons` to be a multiple of 32 with the default bitset dtype. `pop_size` includes both training and evaluation individuals; use at least two training individuals and one evaluation individual.

For a full SATR-EC run, for example:

```bash
python ec_satr_sweep.py task=humanoid \
  es_conf.update_method=signal_adaptive_trust_region \
  es_conf.scale_factor=0.5 es_conf.lr=0.15 network_type=ConnSNN_Optim
```

Configuration uses OmegaConf dot-list arguments. Defaults include population 10,240, evaluation size 128, 2,000 generations and 1,000 steps per episode. These settings can require substantial time and GPU memory. Set `seed`, `total_generations`, `es_conf.pop_size`, `es_conf.eval_size`, `episode_conf.max_episode_length`, and `eval_episode_conf.max_episode_length` as needed. Brax physics can vary numerically even with a fixed seed.

## Methods and entry points

| Script | Purpose |
| --- | --- |
| `ec_satr_sweep.py` | SATR-EC; use `es_conf.update_method=identity` for the EC update |
| `ec_tr_sweep.py` | Fixed trust-region EC baseline |
| `ec_sweep.py`, `ec.py` | EC baselines |
| `es_dense_sweep_stable.py`, `es_dense.py` | Dense evolution-strategy baselines |

The main script supports `ConnSNN_Optim` (bitset) and `ConnSNN` (unpacked). For example, change `network_type=ConnSNN` to compare implementations. Hopper, Walker2d and Humanoid can be selected using `task=hopper`, `task=walker2d` or `task=humanoid`. Experiment settings should be matched to the paper before interpreting results.

## Outputs

W&B logs `fitness` (training return), `eval_fitness`, `sparsity`, and training environment step counts. Evaluation is deterministic with respect to connectivity (`p > 0.5`) and uses a separate environment pool. Checkpoints are written every `save_every` generations under `models/{project_name}/{run_name}/`. Set `save_every` larger than a short test to skip checkpointing.

## Repository layout

- `networks/`: connectivity and dense network implementations and bitset operations.
- `utils/`: shared utilities.
- `conf_experiment/`: example experiment configuration.
- `requirements.txt`: Python dependencies; JAX must be installed separately for the target hardware.

## Citation and license

Please cite the associated paper when using this code. Citation and license details have not yet been supplied in this repository.
