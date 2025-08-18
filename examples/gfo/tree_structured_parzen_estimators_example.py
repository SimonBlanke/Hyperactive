"""
Tree-structured Parzen Estimators Example - Bayesian Optimization with TPE

Tree-structured Parzen Estimators (TPE) is a Bayesian optimization algorithm
that models the distribution of good and bad parameter configurations separately.
It uses these models to suggest new configurations that maximize expected
improvement, making it highly efficient for expensive function evaluations.

Characteristics:
- Bayesian optimization with probabilistic surrogate models
- Separate modeling of good and bad parameter regions
- Expected improvement acquisition function
- Efficient for expensive function evaluations
- Handles mixed parameter types (continuous, discrete, categorical)
"""

import numpy as np
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from hyperactive.experiment.integrations import SklearnCvExperiment
from hyperactive.opt.gfo import TreeStructuredParzenEstimators

# Load dataset
X, y = load_wine(return_X_y=True)
print(f"Dataset: Wine classification ({X.shape[0]} samples, {X.shape[1]} features)")

# Create experiment
estimator = RandomForestClassifier(random_state=42)
experiment = SklearnCvExperiment(estimator=estimator, X=X, y=y, cv=3)

# Define search space
search_space = {
    "n_estimators": list(range(10, 201, 10)),  # Discrete integer values
    "max_depth": list(range(1, 21)),  # Discrete integer values
    "min_samples_split": list(range(2, 21)),  # Discrete integer values
    "min_samples_leaf": list(range(1, 11)),  # Discrete integer values
}

# Configure Tree-structured Parzen Estimators
warm_start_points = [
    {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
    }
]

optimizer = TreeStructuredParzenEstimators(
    search_space=search_space,
    n_iter=15,
    random_state=42,
    initialize={"warm_start": warm_start_points},
    experiment=experiment,
)

# Run optimization
# TPE builds probabilistic models P(x|y) where x are parameters and y are outcomes
# It maintains separate models for good (y > threshold) and bad (y <= threshold) trials
# New configurations are selected to maximize the ratio of good to bad model densities
# This Bayesian approach efficiently guides search toward promising parameter regions
best_params = optimizer.solve()

# Results
print("\n=== Results ===")
print(f"Best parameters: {best_params}")
print("Tree-structured Parzen Estimators optimization completed successfully")
