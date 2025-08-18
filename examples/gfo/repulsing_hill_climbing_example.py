"""
Repulsing Hill Climbing Example - Multi-Point Local Search with Memory

Repulsing Hill Climbing extends hill climbing by maintaining memory of previously
visited good solutions and using this information to repel the search away from
already explored regions. This helps discover multiple local optima and provides
better coverage of the search space than traditional hill climbing.

Characteristics:
- Memory-based search with repulsion mechanism
- Discovers multiple local optima through exploration
- Avoids re-exploring already visited good regions
- Better search space coverage than standard hill climbing
- Effective for problems with multiple attractive regions
"""

import numpy as np
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from hyperactive.experiment.integrations import SklearnCvExperiment
from hyperactive.opt.gfo import RepulsingHillClimbing

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

# Configure Repulsing Hill Climbing
# Multiple warm start points can help demonstrate the repulsion mechanism
warm_start_points = [
    {"n_estimators": 50, "max_depth": 5, "min_samples_split": 10, "min_samples_leaf": 5}
]

optimizer = RepulsingHillClimbing(
    search_space=search_space,
    n_iter=40,
    random_state=42,
    initialize={"warm_start": warm_start_points},
    experiment=experiment
)

# Run optimization
# Repulsing hill climbing maintains a memory of good solutions found
# and uses repulsion forces to push the search away from these areas
# This encourages exploration of new regions and discovery of multiple optima
# The repulsion mechanism helps avoid cycling between the same local optima
best_params = optimizer.solve()

# Results
print("\n=== Results ===")
print(f"Best parameters: {best_params}")
print("Repulsing hill climbing optimization completed successfully")
