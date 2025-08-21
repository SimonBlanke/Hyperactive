"""
GPOptimizer Example - Gaussian Process Bayesian Optimization

The GPOptimizer uses Gaussian Processes to model the objective function and
select promising parameter configurations. It's particularly effective for
expensive function evaluations and provides uncertainty estimates.

Characteristics:
- Bayesian optimization with Gaussian Process surrogate model
- Balances exploration (high uncertainty) and exploitation (high mean)
- Works well with mixed parameter types
- Provides uncertainty quantification
- Efficient for expensive objective functions
- Can handle constraints and noisy observations
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

from hyperactive.experiment.integrations import SklearnCvExperiment
from hyperactive.opt.optuna import GPOptimizer


# Gaussian Process Bayesian Optimization:
#
# 1. Surrogate Model:
#    - GP models f(x) ~ N(μ(x), σ²(x))
#    - μ(x): predicted mean (expected objective value)
#    - σ²(x): predicted variance (uncertainty estimate)
#
# 2. Acquisition Function:
#    - Balances exploration vs exploitation
#    - Common choices: Expected Improvement (EI), Upper Confidence Bound (UCB)
#    - Selects next point to evaluate: x_next = argmax acquisition(x)
#
# 3. Iterative Process:
#    - Fit GP to observed data (x_i, f(x_i))
#    - Optimize acquisition function to find x_next
#    - Evaluate f(x_next)
#    - Update dataset and repeat
#
# 4. Key Advantages:
#    - Uncertainty-aware: explores uncertain regions
#    - Sample efficient: good for expensive evaluations
#    - Principled: grounded in Bayesian inference


# === GPOptimizer Example ===
# Gaussian Process Bayesian Optimization


# Load dataset - classification problem
X, y = load_breast_cancer(return_X_y=True)
print(
    f"Dataset: Breast cancer classification ({X.shape[0]} samples, {X.shape[1]} features)"
)

# Create experiment
estimator = SVC(random_state=42)
experiment = SklearnCvExperiment(estimator=estimator, X=X, y=y, cv=5)

# Define search space - mixed parameter types
param_space = {
    "C": (0.01, 100),  # Continuous - regularization
    "kernel": ["rbf", "poly", "sigmoid"],  # Categorical
}

# Search Space (Mixed parameter types):
# for param, space in param_space.items():
#   print(f"  {param}: {space}")

# Configure GPOptimizer
optimizer = GPOptimizer(
    param_space=param_space,
    n_trials=5,
    random_state=42,
    experiment=experiment,
    n_startup_trials=5,  # Random initialization before GP modeling
    deterministic_objective=False,  # Set True if objective is noise-free
)

# GPOptimizer Configuration:
# n_trials: configured above
# n_startup_trials: random initialization
# deterministic_objective: configures noise handling
# Acquisition function: Expected Improvement (default)

# Run optimization
# Running GP-based optimization...
best_params = optimizer.solve()

# Results
print("\n=== Results ===")
print(f"Best parameters: {best_params}")
print(f"Best score: {optimizer.best_score_:.4f}")
print()

# GP Optimization Phases:
#
# Phase 1 (Trials 1-8): Random Exploration
#  Random sampling for initial GP training data
#  Builds diverse set of observations
#  No model assumptions yet

# Phase 2 (Trials 9-25): GP-guided Search
#  GP model learns from observed data
#  Acquisition function balances:
#   - Exploitation: areas with high predicted performance
#   - Exploration: areas with high uncertainty
#  Sequential decision making with uncertainty

# GP Model Characteristics:
#  Handles mixed parameter types (continuous, discrete, categorical)
#  Provides uncertainty estimates for all predictions
#  Automatically balances exploration vs exploitation
#  Sample efficient - good for expensive evaluations
#  Can incorporate prior knowledge through mean/kernel functions

# Acquisition Function Behavior:
#  High mean + low variance → exploitation
#  Low mean + high variance → exploration
#  Balanced trade-off prevents premature convergence
#  Adapts exploration strategy based on observed data

# Best Use Cases:
#  Expensive objective function evaluations
#  Small to medium parameter spaces (< 20 dimensions)
#  When uncertainty quantification is valuable
#  Mixed parameter types (continuous + categorical)
#  Noisy objective functions (with appropriate kernel)

# Limitations:
#  Computational cost grows with number of observations
#  Hyperparameter tuning for GP kernel
#  May struggle in very high dimensions
#  Assumes some smoothness in objective function

# Comparison with TPESampler:
# GPOptimizer advantages:
#   + Principled uncertainty quantification
#   + Better for expensive evaluations
#   + Can handle constraints naturally
#
# TPESampler advantages:
#   + Faster computation
#   + Better scalability to high dimensions
#   + More robust hyperparameter defaults
