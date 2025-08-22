"""
GridOptimizer Example - Exhaustive Grid Search

The GridOptimizer performs exhaustive search over a discretized parameter grid.
It systematically evaluates every combination of specified parameter values,
ensuring complete coverage but potentially requiring many evaluations.

Characteristics:
- Exhaustive search over predefined parameter grids
- Systematic and reproducible exploration
- Guarantees finding the best combination within the grid
- No learning or adaptation
- Best for small, discrete parameter spaces
- Interpretable and deterministic results
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

from hyperactive.experiment.integrations import SklearnCvExperiment
from hyperactive.opt.optuna import GridOptimizer


# Grid Search Methodology:
#
# 1. Parameter Discretization:
#    - Each continuous parameter divided into discrete levels
#    - Categorical parameters use all specified values
#    - Creates n₁ × n₂ × ... × nₖ total combinations
#
# 2. Systematic Evaluation:
#    - Every combination evaluated exactly once
#    - No randomness or learning involved
#    - Order of evaluation is deterministic
#
# 3. Optimality Guarantees:
#    - Finds global optimum within the discrete grid
#    - Quality depends on grid resolution
#    - May miss optimal values between grid points
#
# 4. Computational Complexity:
#    - Exponential growth with number of parameters
#    - Curse of dimensionality for many parameters
#    - Embarrassingly parallel


# === GridOptimizer Example ===
# Exhaustive Grid Search


# Load dataset - simple classification
X, y = load_iris(return_X_y=True)
print(f"Dataset: Iris classification ({X.shape[0]} samples, {X.shape[1]} features)")

# Create experiment
estimator = KNeighborsClassifier()
experiment = SklearnCvExperiment(estimator=estimator, X=X, y=y, cv=3)

# Define search space - DISCRETE values only for grid search
param_space = {
    "n_neighbors": [1, 3, 5, 7, 11, 15, 21],  # 7 values
    "weights": ["uniform", "distance"],  # 2 values
    "metric": ["euclidean", "manhattan", "minkowski"],  # 3 values
    "p": [1, 2],  # Only relevant for minkowski metric     # 2 values
}

# Total combinations: 7 × 2 × 3 × 2 = 84 combinations
total_combinations = 1
for param, values in param_space.items():
    total_combinations *= len(values)

# Search Space (Discrete grids only):
# for param, values in param_space.items():
#   print(f"  {param}: {values} ({len(values)} values)")
# Total combinations: calculated above

# Configure GridOptimizer
optimizer = GridOptimizer(
    param_space=param_space,
    n_trials=total_combinations,  # Will evaluate all combinations
    random_state=42,  # For deterministic ordering
    experiment=experiment,
)

# GridOptimizer Configuration:
# n_trials: matches total combinations
# search_space: automatically derived from param_space
# Systematic evaluation of every combination

# Run optimization
# Running exhaustive grid search...
best_params = optimizer.solve()

# Results
print("\n=== Results ===")
print(f"Best parameters: {best_params}")
print(f"Best score: {optimizer.best_score_:.4f}")
print()

# Grid Search Characteristics:
#
#  Exhaustive Coverage:
#   - Evaluated all parameter combinations
#   - Guaranteed to find best configuration within grid
#   - No risk of missing good regions

#  Reproducibility:
#   - Same grid → same results every time
#   - Deterministic evaluation order
#   - No randomness or hyperparameters

#  Interpretability:
#   - Easy to understand methodology
#   - Clear relationship between grid density and accuracy
#   - Results easily visualized and analyzed

# Grid Design Considerations:
#
# Parameter Value Selection:
#  Include reasonable ranges for each parameter
#  Use domain knowledge to choose meaningful values
#  Consider logarithmic spacing for scale-sensitive parameters
#  Start coarse, then refine around promising regions
#
# Computational Budget:
#  Balance grid density with available compute
#  Consider parallel evaluation to speed up
#  Use coarse grids for initial exploration
#
# Best Use Cases:
#  Small parameter spaces (< 6 parameters)
#  Discrete/categorical parameters
#  When exhaustive evaluation is feasible
#  Baseline comparison for other methods
#  When interpretability is crucial
#  Parallel computing environments
#
# Limitations:
#  Exponential scaling with parameter count
#  May miss optimal values between grid points
#  Inefficient for continuous parameters
#  No adaptive learning or focusing
#  Can waste evaluations in clearly bad regions
#
# Grid Search vs Other Methods:
#
# vs Random Search:
#   + Systematic coverage guarantee
#   + Reproducible results
#   - Exponential scaling
#   - Less efficient in high dimensions
#
# vs Bayesian Optimization:
#   + No assumptions about objective function
#   + Guaranteed to find grid optimum
#   - Much less sample efficient
#   - No learning from previous evaluations
