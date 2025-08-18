"""
Downhill Simplex Example - Nelder-Mead Optimization

Downhill Simplex (Nelder-Mead) maintains a simplex of n+1 points in n-dimensional
space and iteratively transforms this simplex through reflection, expansion,
contraction, and shrinkage operations. It's a robust derivative-free method
that's particularly effective for low-dimensional continuous optimization.

Characteristics:
- Maintains a simplex of n+1 points in n-dimensional space
- Uses geometric transformations (reflect, expand, contract, shrink)
- Derivative-free optimization method
- Good convergence properties for smooth functions
- Effective for low to moderate dimensional problems
"""

import numpy as np
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from hyperactive.experiment.integrations import SklearnCvExperiment
from hyperactive.opt.gfo import DownhillSimplexOptimizer

# Load dataset
X, y = load_wine(return_X_y=True)
print(f"Dataset: Wine classification ({X.shape[0]} samples, {X.shape[1]} features)")

# Create experiment
estimator = RandomForestClassifier(random_state=42)
experiment = SklearnCvExperiment(estimator=estimator, X=X, y=y, cv=3)

# Define search space
search_space = {
    "n_estimators": list(range(10, 201, 15)),    # Discrete integer values
    "max_depth": list(range(1, 21)),             # Discrete integer values
    "min_samples_split": list(range(2, 21)),     # Discrete integer values
    "min_samples_leaf": list(range(1, 11)),      # Discrete integer values
}

# Configure Downhill Simplex Optimizer
warm_start_points = [
    {"n_estimators": 100, "max_depth": 10, "min_samples_split": 5, "min_samples_leaf": 2}
]

optimizer = DownhillSimplexOptimizer(
    search_space=search_space,
    n_iter=35,
    random_state=42,
    initialize={"warm_start": warm_start_points},
    experiment=experiment
)

# Run optimization
# Downhill simplex maintains a simplex (triangle in 2D, tetrahedron in 3D, etc.)
# and applies geometric transformations: reflection (flip worst point),
# expansion (extend good moves), contraction (reduce bad moves),
# and shrinkage (reduce entire simplex) to iteratively improve the simplex
best_params = optimizer.solve()

# Results
print("\n=== Results ===")
print(f"Best parameters: {best_params}")
print("Downhill simplex optimization completed successfully")