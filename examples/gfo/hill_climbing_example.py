"""
Hill Climbing Example - Local Search Optimization

Hill climbing is a local search algorithm that starts from a random point and
iteratively moves to neighboring solutions with better objective values. It's
a simple but effective optimization strategy that can quickly find good local
optima, especially when started from reasonable initial points.

Characteristics:
- Local search strategy (moves to better neighbors)
- Fast convergence to local optima
- Simple and interpretable algorithm
- Good for continuous optimization problems
- Can get stuck in local maxima
"""

import numpy as np
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from hyperactive.experiment.integrations import SklearnCvExperiment
from hyperactive.opt.gfo import HillClimbing

# Load dataset
X, y = load_wine(return_X_y=True)
print(f"Dataset: Wine classification ({X.shape[0]} samples, {X.shape[1]} features)")

# Create experiment
estimator = RandomForestClassifier(random_state=42)
experiment = SklearnCvExperiment(estimator=estimator, X=X, y=y, cv=3)

# Define search space - discrete values for hill climbing
search_space = {
    "n_estimators": list(range(10, 201)),       # Discrete integer values
    "max_depth": list(range(1, 21)),            # Discrete integer values
    "min_samples_split": list(range(2, 21)),    # Discrete integer values
    "min_samples_leaf": list(range(1, 11)),     # Discrete integer values
}

# Configure HillClimbing optimizer
# Starting from a reasonable initial point can improve performance
warm_start_points = [
    {"n_estimators": 100, "max_depth": 10, "min_samples_split": 5, "min_samples_leaf": 2}
]

optimizer = HillClimbing(
    search_space=search_space,
    n_iter=40,
    random_state=42,
    initialize={"warm_start": warm_start_points},
    experiment=experiment
)

# Run optimization
# Hill climbing explores the neighborhood of current best solution
# and moves to better solutions when found, creating a local search pattern
best_params = optimizer.solve()

# Results
print("\n=== Results ===")
print(f"Best parameters: {best_params}")
print("Hill climbing optimization completed successfully")
