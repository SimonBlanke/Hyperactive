"""
Forest Optimizer Example - Random Forest-Based Surrogate Optimization

Forest Optimizer uses Random Forest models as surrogate functions to approximate
the expensive objective function. It leverages the Random Forest's ability to
capture non-linear relationships and provide uncertainty estimates to guide
the search toward promising regions of the parameter space.

Characteristics:
- Random Forest surrogate model for objective function approximation
- Uncertainty-aware search through forest prediction variance
- Good for capturing non-linear parameter relationships
- Robust to noise and handles mixed parameter types well
- Efficient for moderately expensive function evaluations
"""

import numpy as np
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from hyperactive.experiment.integrations import SklearnCvExperiment
from hyperactive.opt.gfo import ForestOptimizer

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

# Configure Forest Optimizer
warm_start_points = [
    {"n_estimators": 80, "max_depth": 8, "min_samples_split": 5, "min_samples_leaf": 3}
]

optimizer = ForestOptimizer(
    search_space=search_space,
    n_iter=35,
    random_state=42,
    initialize={"warm_start": warm_start_points},
    experiment=experiment
)

# Run optimization
# Forest optimizer builds Random Forest surrogate models from observed data
# It uses the forest predictions to estimate objective values and uncertainties
# New points are selected based on acquisition functions that balance
# predicted performance with prediction uncertainty (exploration vs exploitation)
best_params = optimizer.solve()

# Results
print("\n=== Results ===")
print(f"Best parameters: {best_params}")
print("Forest optimization completed successfully")