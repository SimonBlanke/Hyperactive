"""
CmaEsOptimizer Example - Covariance Matrix Adaptation Evolution Strategy

CMA-ES is a powerful evolution strategy particularly effective for continuous
optimization problems. It adapts both the mean and covariance matrix of a
multivariate normal distribution to efficiently explore the parameter space.

Characteristics:
- Excellent for continuous parameter optimization
- Adapts search distribution shape and orientation
- Self-adaptive step size control
- Handles ill-conditioned problems well
- Does not work with categorical parameters
- Requires 'cmaes' package: pip install cmaes

Note: This example includes a fallback if 'cmaes' package is not installed.
"""

import numpy as np
from sklearn.datasets import make_regression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

from hyperactive.experiment.integrations import SklearnCvExperiment
from hyperactive.opt.optuna import CmaEsOptimizer


# CMA-ES Algorithm Theory:
# 1. Maintains a multivariate normal distribution N(μ, σ²C)
#    - μ: mean vector (center of search)
#    - σ: step size (global scaling)
#    - C: covariance matrix (shape and orientation)
#
# 2. In each generation:
#    - Sample λ offspring from N(μ, σ²C)
#    - Evaluate all offspring
#    - Select μ best solutions
#    - Update μ, σ, and C based on selected solutions
#
# 3. Adaptive features:
#    - Covariance matrix learns correlations between parameters
#    - Step size adapts to local landscape
#    - Handles rotated/scaled problems efficiently


# === CmaEsOptimizer Example ===
# Covariance Matrix Adaptation Evolution Strategy

# Check if cmaes is available
try:
    import cmaes

    cmaes_available = True
    print(" CMA-ES package is available")
except ImportError:
    cmaes_available = False
    print("⚠ CMA-ES package not available. Install with: pip install cmaes")
    print("  This example will demonstrate the interface but may fail at runtime.")
print()

cmaes_theory()

# Create a continuous optimization problem
X, y = make_regression(n_samples=200, n_features=10, noise=0.1, random_state=42)
print(f"Dataset: Synthetic regression ({X.shape[0]} samples, {X.shape[1]} features)")

# Create experiment - neural network with continuous parameters
estimator = MLPRegressor(random_state=42, max_iter=1000)
experiment = SklearnCvExperiment(
    estimator=estimator, X=X, y=y, cv=3, scoring="neg_mean_squared_error"
)

# Define search space - ONLY continuous parameters (CMA-ES limitation)
param_space = {
    "alpha": (1e-6, 1e-1),  # L2 regularization
    "learning_rate_init": (1e-4, 1e-1),  # Initial learning rate
    "beta_1": (0.8, 0.99),  # Adam beta1 parameter
    "beta_2": (0.9, 0.999),  # Adam beta2 parameter
    "epsilon": (1e-9, 1e-6),  # Adam epsilon parameter
    # Note: No categorical parameters - CMA-ES doesn't support them
}

# Search Space (Continuous parameters only):
# for param, space in param_space.items():
#   print(f"  {param}: {space}")
# Note: CMA-ES only works with continuous parameters
# For mixed parameter types, consider TPESampler or GPOptimizer

# Configure CmaEsOptimizer
optimizer = CmaEsOptimizer(
    param_space=param_space,
    n_trials=20,
    random_state=42,
    experiment=experiment,
    sigma0=0.2,  # Initial step size (exploration vs exploitation)
    n_startup_trials=5,  # Random trials before CMA-ES starts
)

# CmaEsOptimizer Configuration:
# n_trials: configured above
# sigma0: initial step size
# n_startup_trials: random trials before CMA-ES starts
# Adaptive covariance matrix will be learned during optimization

if not cmaes_available:
    print("⚠ Skipping optimization due to missing 'cmaes' package")
    print("Install with: pip install cmaes")
    return None, None

# Run optimization
# Running CMA-ES optimization...
try:
    best_params = optimizer.solve()

    # Results
    print("\n=== Results ===")
    print(f"Best parameters: {best_params}")
    print(f"Best score: {optimizer.best_score_:.4f}")
    print()

except ImportError as e:
    print(f"CMA-ES failed: {e}")
    print("Install the required package: pip install cmaes")
    return None, None

# CMA-ES Behavior Analysis:
# Evolution of search distribution:
#  Initial: Spherical distribution (σ₀ * I)
#  Early trials: Random exploration to gather information
#  Mid-trials: Covariance matrix learns parameter correlations
#  Later trials: Focused search along principal component directions

# Adaptive Properties:
#  Step size (σ) adapts to local topology
#  Covariance matrix (C) learns parameter interactions
#  Mean vector (μ) tracks promising regions
#  Handles ill-conditioned and rotated problems

# Best Use Cases:
#  Continuous optimization problems
#  Parameters with potential correlations
#  Non-convex, multimodal functions
#  When gradient information is unavailable
#  Medium-dimensional problems (2-40 parameters)

# Limitations:
#  Only continuous parameters (no categorical/discrete)
#  Requires additional 'cmaes' package
#  Can be slower than TPE for simple problems
#  Memory usage grows with parameter dimension
