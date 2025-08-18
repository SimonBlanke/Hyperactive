"""
Sklearn Random Search Example - Native Sklearn RandomizedSearchCV Integration

This example demonstrates using Hyperactive's sklearn backend with
RandomizedSearchCV. This provides access to sklearn's mature random search
implementation with support for probability distributions and efficient
sampling strategies.

Characteristics:
- Direct integration with sklearn's RandomizedSearchCV
- Support for probability distributions (not just discrete lists)
- Efficient random sampling from continuous and discrete spaces
- Sklearn-native cross-validation and scoring
- Good baseline performance with minimal configuration
"""

import numpy as np
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from hyperactive.experiment.integrations import SklearnCvExperiment
from hyperactive.opt import RandomSearchSk

# Load dataset
X, y = load_wine(return_X_y=True)
print(f"Dataset: Wine classification ({X.shape[0]} samples, {X.shape[1]} features)")

# Create experiment
estimator = RandomForestClassifier(random_state=42)
experiment = SklearnCvExperiment(estimator=estimator, X=X, y=y, cv=3)

# Define parameter distributions for random sampling
# RandomizedSearchCV can sample from lists (uniform) or distributions
param_distributions = {
    "n_estimators": list(range(10, 201)),        # Discrete list (uniform sampling)
    "max_depth": list(range(1, 21)),             # Discrete integer range
    "min_samples_split": list(range(2, 21)),     # Discrete integer range
    "min_samples_leaf": list(range(1, 11)),      # Discrete integer range
    "max_features": ["sqrt", "log2", None],      # Discrete categorical choices
    "bootstrap": [True, False],                  # Binary choice
}

# Configure Sklearn Random Search
optimizer = RandomSearchSk(
    param_distributions=param_distributions,
    n_iter=30,
    random_state=42,
    experiment=experiment
)

# Run optimization
# The sklearn backend uses RandomizedSearchCV internally, which efficiently
# samples from the defined parameter distributions without requiring
# explicit enumeration of all possible values
best_params = optimizer.solve()

# Results
print("\n=== Results ===")
print(f"Best parameters: {best_params}")
print(f"Best score: {optimizer.best_score_:.4f}")
print(f"Randomly sampled 30 parameter combinations")
