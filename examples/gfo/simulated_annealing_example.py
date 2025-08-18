"""
Simulated Annealing Example - Probabilistic Hill Climbing

Simulated Annealing is inspired by the metallurgical process of annealing, where
metals are heated and slowly cooled to reduce defects. This algorithm starts with
high "temperature" allowing random moves, then gradually cools down to focus on
local improvements. The cooling schedule allows escaping local optima early on.

Characteristics:
- Probabilistic acceptance of worse solutions (temperature-dependent)
- Gradually reduces exploration over time (cooling schedule)
- Can escape local optima unlike pure hill climbing
- Good balance between exploration and exploitation
- Single-point search with memory of current solution
"""

import numpy as np
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from hyperactive.experiment.integrations import SklearnCvExperiment
from hyperactive.opt.gfo import SimulatedAnnealing

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

# Configure Simulated Annealing
# Start with good initial point to demonstrate cooling behavior
warm_start_points = [
    {"n_estimators": 100, "max_depth": 10, "min_samples_split": 5, "min_samples_leaf": 2}
]

optimizer = SimulatedAnnealing(
    search_space=search_space,
    n_iter=40,
    random_state=42,
    initialize={"warm_start": warm_start_points},
    experiment=experiment
)

# Run optimization
# Simulated annealing uses a cooling schedule to gradually reduce temperature
# Early iterations: high probability of accepting worse solutions (exploration)
# Later iterations: low probability of accepting worse solutions (exploitation)
best_params = optimizer.solve()

# Results
print("\n=== Results ===")
print(f"Best parameters: {best_params}")
print("Simulated annealing optimization completed successfully")
