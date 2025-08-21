"""
Sklearn Grid Search Example - Native Sklearn GridSearchCV Integration

This example demonstrates using Hyperactive's sklearn backend that provides
a direct interface to scikit-learn's GridSearchCV. This approach leverages
sklearn's mature and optimized grid search implementation while maintaining
compatibility with Hyperactive's unified interface.

Characteristics:
- Direct integration with sklearn's GridSearchCV
- Optimized for sklearn estimators and pipelines
- Supports sklearn's built-in cross-validation strategies
- Familiar sklearn-style parameter specification
- Efficient parallel execution when n_jobs > 1
"""

import numpy as np
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from hyperactive.experiment.integrations import SklearnCvExperiment
from hyperactive.opt import GridSearchSk

# Load dataset
X, y = load_wine(return_X_y=True)
print(f"Dataset: Wine classification ({X.shape[0]} samples, {X.shape[1]} features)")

# Create experiment
estimator = RandomForestClassifier(random_state=42)
experiment = SklearnCvExperiment(estimator=estimator, X=X, y=y, cv=3)

# Define parameter grid in sklearn style
# Using lists of discrete values for exhaustive search
param_grid = {
    "n_estimators": [10, 50, 100, 150],          # Discrete values for grid
    "max_depth": [5, 10, 15, None],              # Include None for unlimited
    "min_samples_split": [2, 5, 10],             # Small discrete set
    "max_features": ["sqrt", "log2", None],      # Feature selection strategies
    "bootstrap": [True, False],                  # Bootstrap sampling options
}

# Calculate total parameter combinations
total_combinations = 1
for param_values in param_grid.values():
    total_combinations *= len(param_values)

print(f"Total parameter combinations to evaluate: {total_combinations}")

# Configure Sklearn Grid Search
optimizer = GridSearchSk(
    param_grid=param_grid,
    experiment=experiment
)

# Run optimization
# The sklearn backend uses GridSearchCV internally, providing
# optimized grid search with sklearn's efficient implementation
best_params = optimizer.solve()

# Results
print("\n=== Results ===")
print(f"Best parameters: {best_params}")
print(f"Best score: {optimizer.best_score_:.4f}")
print(f"Exhaustively evaluated {total_combinations} combinations")
