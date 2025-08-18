"""
Stochastic Hill Climbing Example - Randomized Local Search

Stochastic Hill Climbing extends basic hill climbing by introducing randomness
in the neighbor selection process. Instead of always choosing the best neighbor,
it probabilistically selects from improving neighbors, which helps avoid getting
stuck in plateaus and provides better exploration of the search space.

Characteristics:
- Probabilistic neighbor selection (not always greedy)
- Better plateau handling than deterministic hill climbing
- Maintains local search efficiency with improved exploration
- Good balance between exploitation and local exploration
- Simpler than simulated annealing but more flexible than pure hill climbing
"""

import numpy as np
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from hyperactive.experiment.integrations import SklearnCvExperiment
from hyperactive.opt.gfo import StochasticHillClimbing

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

# Configure Stochastic Hill Climbing
warm_start_points = [
    {"n_estimators": 80, "max_depth": 8, "min_samples_split": 5, "min_samples_leaf": 3}
]

optimizer = StochasticHillClimbing(
    search_space=search_space,
    n_iter=40,
    random_state=42,
    initialize={"warm_start": warm_start_points},
    experiment=experiment
)

# Run optimization
# Stochastic hill climbing uses randomness in neighbor selection
# Instead of always picking the best neighbor, it probabilistically
# selects from improving neighbors, helping to navigate plateaus
# and avoid getting trapped in shallow local optima
best_params = optimizer.solve()

# Results
print("\n=== Results ===")
print(f"Best parameters: {best_params}")
print("Stochastic hill climbing optimization completed successfully")
