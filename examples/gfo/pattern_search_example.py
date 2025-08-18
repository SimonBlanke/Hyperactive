"""
Pattern Search Example - Direct Search with Systematic Patterns

Pattern Search is a direct search method that explores the search space using
a systematic pattern of points around the current best solution. It's particularly
robust for noisy functions and doesn't require gradient information, making it
suitable for black-box optimization problems.

Characteristics:
- Systematic pattern-based exploration around current solution
- Robust to function noise and discontinuities
- Derivative-free direct search method
- Adaptive step size based on search success
- Good for black-box optimization problems
"""

import numpy as np
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from hyperactive.experiment.integrations import SklearnCvExperiment
from hyperactive.opt.gfo import PatternSearch

# Load dataset
X, y = load_wine(return_X_y=True)
print(f"Dataset: Wine classification ({X.shape[0]} samples, {X.shape[1]} features)")

# Create experiment
estimator = RandomForestClassifier(random_state=42)
experiment = SklearnCvExperiment(estimator=estimator, X=X, y=y, cv=3)

# Define search space
search_space = {
    "n_estimators": list(range(10, 201, 10)),    # Discrete integer values
    "max_depth": list(range(1, 21)),             # Discrete integer values
    "min_samples_split": list(range(2, 21)),     # Discrete integer values
    "min_samples_leaf": list(range(1, 11)),      # Discrete integer values
}

# Configure Pattern Search
warm_start_points = [
    {"n_estimators": 80, "max_depth": 8, "min_samples_split": 8, "min_samples_leaf": 3}
]

optimizer = PatternSearch(
    search_space=search_space,
    n_iter=35,
    random_state=42,
    initialize={"warm_start": warm_start_points},
    experiment=experiment
)

# Run optimization
# Pattern search systematically evaluates points in a pattern around
# the current best solution. If a better point is found, it becomes the new
# center and the pattern is applied again. If no improvement is found,
# the step size is reduced and the search continues with finer resolution
best_params = optimizer.solve()

# Results
print("\n=== Results ===")
print(f"Best parameters: {best_params}")
print("Pattern search optimization completed successfully")
