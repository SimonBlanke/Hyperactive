"""
Random Search Example - Pure Random Sampling

Random search is one of the simplest optimization algorithms that samples parameters
randomly from the defined search space. Despite its simplicity, it's surprisingly
effective for many hyperparameter optimization tasks and serves as an excellent
baseline for comparison with more sophisticated algorithms.

Characteristics:
- No learning or adaptation between trials
- Uniform sampling from the search space
- Excellent baseline for comparison
- Works well in high-dimensional spaces
- Parallel-friendly (no dependencies between trials)
"""

import numpy as np
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from hyperactive.experiment.integrations import SklearnCvExperiment
from hyperactive.opt.gfo import RandomSearch

# Load dataset
X, y = load_wine(return_X_y=True)
print(f"Dataset: Wine classification ({X.shape[0]} samples, {X.shape[1]} features)")

# Create experiment
estimator = RandomForestClassifier(random_state=42)
experiment = SklearnCvExperiment(estimator=estimator, X=X, y=y, cv=3)

# Define search space
search_space = {
    "n_estimators": list(range(10, 201, 10)),    # Discrete integer values (reduced for speed)
    "max_depth": list(range(1, 21)),             # Discrete integer values
    "min_samples_split": list(range(2, 21)),     # Discrete integer values
    "min_samples_leaf": list(range(1, 11)),      # Discrete integer values
}

# Configure RandomSearch optimizer
optimizer = RandomSearch(
    search_space=search_space,
    n_iter=30,
    random_state=42,
    experiment=experiment
)

# Run optimization
# Random search samples each parameter independently and uniformly
# from its defined range or choices, making it simple but effective
best_params = optimizer.solve()

# Results
print("\n=== Results ===")
print(f"Best parameters: {best_params}")
print("Optimization completed successfully")
