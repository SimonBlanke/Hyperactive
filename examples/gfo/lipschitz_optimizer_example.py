"""
Lipschitz Optimizer Example - Lipschitz Constant-Based Global Optimization

Lipschitz Optimization assumes the objective function satisfies a Lipschitz
condition (bounded rate of change) and uses this information to eliminate
regions that cannot contain the global optimum. This provides theoretical
convergence guarantees and efficient global optimization.

Characteristics:
- Uses Lipschitz constant to bound function behavior
- Theoretical convergence guarantees to global optimum
- Efficient elimination of non-promising regions
- No local optima trapping due to global search strategy
- Particularly effective for functions with known smoothness properties
"""

import numpy as np
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from hyperactive.experiment.integrations import SklearnCvExperiment
from hyperactive.opt.gfo import LipschitzOptimizer

# Load dataset
X, y = load_wine(return_X_y=True)
print(f"Dataset: Wine classification ({X.shape[0]} samples, {X.shape[1]} features)")

# Create experiment
estimator = RandomForestClassifier(random_state=42)
experiment = SklearnCvExperiment(estimator=estimator, X=X, y=y, cv=3)

# Define search space
search_space = {
    "n_estimators": list(range(10, 201, 15)),  # Discrete integer values
    "max_depth": list(range(1, 21, 2)),  # Discrete integer values
    "min_samples_split": list(range(2, 21, 2)),  # Discrete integer values
    "min_samples_leaf": list(range(1, 11)),  # Discrete integer values
}

# Configure Lipschitz Optimizer
optimizer = LipschitzOptimizer(
    search_space=search_space, n_iter=15, random_state=42, experiment=experiment
)

# Run optimization
# Lipschitz optimization estimates the Lipschitz constant and uses it to
# construct lower bounds on the objective function. It eliminates regions
# where the lower bound exceeds the current best value, focusing search
# on regions that could potentially contain the global optimum
best_params = optimizer.solve()

# Results
print("\n=== Results ===")
print(f"Best parameters: {best_params}")
print("Lipschitz optimization completed successfully")
