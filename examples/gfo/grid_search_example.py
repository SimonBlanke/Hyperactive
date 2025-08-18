"""
Grid Search Example - Exhaustive Parameter Space Search

Grid search systematically evaluates all possible combinations of parameters
from a predefined discrete grid. While computationally expensive, it guarantees
finding the optimal solution within the defined grid and provides comprehensive
coverage of the parameter space.

Characteristics:
- Exhaustive search over discrete parameter grid
- Guaranteed to find optimal solution in the grid
- Systematic and reproducible results
- Computationally expensive for large grids
- Best for small discrete parameter spaces
"""

import numpy as np
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from hyperactive.experiment.integrations import SklearnCvExperiment
from hyperactive.opt.gfo import GridSearch

# Load dataset
X, y = load_wine(return_X_y=True)
print(f"Dataset: Wine classification ({X.shape[0]} samples, {X.shape[1]} features)")

# Create experiment
estimator = RandomForestClassifier(random_state=42)
experiment = SklearnCvExperiment(estimator=estimator, X=X, y=y, cv=3)

# Define discrete grid - grid search requires explicit discrete values
search_space = {
    "n_estimators": [10, 50, 100, 200],         # Discrete values only
    "max_depth": [5, 10, 15, 20],               # Discrete depth values
    "min_samples_split": [2, 5, 10],            # Small discrete set
    "min_samples_leaf": [1, 2, 5],              # Discrete leaf values
}

# Calculate total combinations
total_combinations = 1
for param_values in search_space.values():
    total_combinations *= len(param_values)

print(f"Total parameter combinations: {total_combinations}")

# Configure Grid Search optimizer
optimizer = GridSearch(
    search_space=search_space,
    n_iter=total_combinations,  # Evaluate all combinations
    random_state=42,
    experiment=experiment
)

# Run optimization
# Grid search will systematically evaluate every possible combination
# of parameters from the defined grid, ensuring complete coverage
best_params = optimizer.solve()

# Results
print("\n=== Results ===")
print(f"Best parameters: {best_params}")
print(f"Evaluated {total_combinations} parameter combinations")
print("Grid search completed successfully")