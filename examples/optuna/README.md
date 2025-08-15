# Optuna Sampler Examples

This directory contains comprehensive examples demonstrating each Optuna sampler available in Hyperactive. Each example shows the sampler's behavior, characteristics, and best use cases.

## Quick Start

Run any example directly:
```bash
python tpe_sampler_example.py
python random_sampler_example.py
# ... etc
```

## Sampler Overview

| Sampler | Type | Best For | Characteristics |
|---------|------|----------|----------------|
| [TPESampler](tpe_sampler_example.py) | Bayesian | General use | Default choice, good balance |
| [RandomSampler](random_sampler_example.py) | Random | Baselines, noisy objectives | Simple, parallel-friendly |
| [CmaEsSampler](cmaes_sampler_example.py) | Evolution Strategy | Continuous optimization | Learns parameter correlations |
| [GPSampler](gp_sampler_example.py) | Bayesian | Expensive evaluations | Uncertainty quantification |
| [GridSampler](grid_sampler_example.py) | Exhaustive | Small spaces | Systematic coverage |
| [NSGAIISampler](nsga_ii_sampler_example.py) | Multi-objective | 2 objectives | Pareto optimization |
| [NSGAIIISampler](nsga_iii_sampler_example.py) | Multi-objective | 3+ objectives | Many-objective problems |
| [QMCSampler](qmc_sampler_example.py) | Quasi-random | Space exploration | Low-discrepancy sequences |

## Detailed Examples

### 1. TPESampler - Tree-structured Parzen Estimator
**File:** `tpe_sampler_example.py`

The default and most popular choice. Uses Bayesian optimization to model good vs bad parameter regions.

```python
from hyperactive.opt.optuna import TPESampler

optimizer = TPESampler(
    param_space=param_space,
    n_trials=50,
    random_state=42,
    n_startup_trials=10,  # Random trials before TPE
    initialize={"warm_start": [good_params]}
)
```

**Best for:** General hyperparameter optimization, mixed parameter types

### 2. RandomSampler - Pure Random Search
**File:** `random_sampler_example.py`

Simple random sampling, surprisingly effective and good baseline.

```python
from hyperactive.opt.optuna import RandomSampler

optimizer = RandomSampler(
    param_space=param_space,
    n_trials=30,
    random_state=42
)
```

**Best for:** Baselines, noisy objectives, high-dimensional spaces

### 3. CmaEsSampler - Covariance Matrix Adaptation
**File:** `cmaes_sampler_example.py`

Evolution strategy that adapts search distribution shape. Requires `pip install cmaes`.

```python
from hyperactive.opt.optuna import CmaEsSampler

optimizer = CmaEsSampler(
    param_space=continuous_params,  # Only continuous!
    n_trials=40,
    sigma0=0.2,  # Initial step size
    random_state=42
)
```

**Best for:** Continuous optimization, parameter correlations

### 4. GPSampler - Gaussian Process Optimization
**File:** `gp_sampler_example.py`

Bayesian optimization with uncertainty quantification.

```python
from hyperactive.opt.optuna import GPSampler

optimizer = GPSampler(
    param_space=param_space,
    n_trials=25,
    n_startup_trials=8,
    deterministic_objective=False
)
```

**Best for:** Expensive evaluations, uncertainty-aware optimization

### 5. GridSampler - Exhaustive Grid Search
**File:** `grid_sampler_example.py`

Systematic evaluation of discrete parameter grids.

```python
from hyperactive.opt.optuna import GridSampler

param_space = {
    "n_neighbors": [1, 3, 5, 7, 11],     # Discrete values only
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan"]
}

optimizer = GridSampler(
    param_space=param_space,
    n_trials=total_combinations
)
```

**Best for:** Small discrete spaces, exhaustive analysis

### 6. NSGAIISampler - Multi-objective (2 objectives)
**File:** `nsga_ii_sampler_example.py`

Multi-objective optimization for two conflicting objectives.

```python
from hyperactive.opt.optuna import NSGAIISampler

# Multi-objective experiment returning [obj1, obj2]
optimizer = NSGAIISampler(
    param_space=param_space,
    n_trials=50,
    population_size=20,
    mutation_prob=0.1,
    crossover_prob=0.9
)
```

**Best for:** Trade-off analysis, accuracy vs complexity

### 7. NSGAIIISampler - Many-objective (3+ objectives)
**File:** `nsga_iii_sampler_example.py`

Many-objective optimization using reference points.

```python
from hyperactive.opt.optuna import NSGAIIISampler

# Many-objective experiment returning [obj1, obj2, obj3, obj4]
optimizer = NSGAIIISampler(
    param_space=param_space,
    n_trials=60,
    population_size=24
)
```

**Best for:** 3+ objectives, complex trade-offs

### 8. QMCSampler - Quasi-Monte Carlo
**File:** `qmc_sampler_example.py`

Low-discrepancy sequences for uniform space filling.

```python
from hyperactive.opt.optuna import QMCSampler

optimizer = QMCSampler(
    param_space=param_space,
    n_trials=32,  # Power of 2 recommended
    qmc_type="sobol",
    scramble=True
)
```

**Best for:** Space exploration, design of experiments

## Common Features

All samplers support:
- **Random state:** `random_state=42` for reproducibility
- **Early stopping:** `early_stopping=10` stop after N trials without improvement
- **Max score:** `max_score=0.99` stop when target reached
- **Warm start:** `initialize={"warm_start": [points]}` initial good points

## Choosing the Right Sampler

### Quick Decision Tree

1. **Multiple objectives?**
   - 2 objectives → NSGAIISampler
   - 3+ objectives → NSGAIIISampler

2. **Single objective:**
   - Need baseline/comparison → RandomSampler
   - Small discrete space → GridSampler  
   - Expensive evaluations → GPSampler
   - Only continuous params → CmaEsSampler
   - Space exploration → QMCSampler
   - General case → **TPESampler** (recommended)

### Computational Budget

- **Low budget (< 50 trials):** RandomSampler, QMCSampler
- **Medium budget (50-200 trials):** TPESampler, GPSampler
- **High budget (200+ trials):** CmaEsSampler, GridSampler

### Parameter Types

- **Mixed types:** TPESampler, GPSampler, RandomSampler
- **Continuous only:** CmaEsSampler
- **Discrete only:** GridSampler

## Advanced Usage

### Combining Samplers

```python
# Phase 1: Initial exploration
qmc_optimizer = QMCSampler(n_trials=20, ...)
initial_results = qmc_optimizer.run()

# Phase 2: Focused optimization
tpe_optimizer = TPESampler(
    n_trials=30,
    initialize={"warm_start": [initial_results]}
)
final_results = tpe_optimizer.run()
```

### Multi-objective Analysis

```python
# For multi-objective problems, you'll typically get multiple solutions
# along the Pareto front. Choose based on your preferences:

solutions = nsga_ii_optimizer.run()
# In practice, you'd analyze the trade-off curve
```

## Dependencies

Most samplers work out of the box. Additional dependencies:
- **CmaEsSampler:** `pip install cmaes`
- All others: Only require `optuna` (included with Hyperactive)

## Performance Tips

1. **Start with TPESampler** for general problems
2. **Use RandomSampler** as baseline comparison
3. **Powers of 2** for QMCSampler trials (32, 64, 128, etc.)
4. **Warm start** with good initial points when available
5. **Early stopping** to avoid wasted evaluations
6. **Random state** for reproducible experiments

## Further Reading

- [Optuna Documentation](https://optuna.readthedocs.io/)
- [Hyperactive Documentation](https://hyperactive.readthedocs.io/)
- [Bayesian Optimization Review](https://arxiv.org/abs/1807.02811)
- [Multi-objective Optimization Survey](https://arxiv.org/abs/1909.04109)