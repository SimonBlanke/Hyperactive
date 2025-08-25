"""
Random Restart Hill Climbing Example - Multi-Start Local Search

Random Restart Hill Climbing performs multiple hill climbing runs from different
random starting points. This strategy overcomes the main limitation of single-point
hill climbing by exploring multiple regions of the search space, significantly
increasing the chances of finding the global optimum.

Characteristics:
- Multiple independent hill climbing runs
- Each restart explores a different region of search space
- Combines local search efficiency with global exploration
- Simple but effective approach for multi-modal problems
- Performance scales with number of restarts and problem difficulty
"""

import numpy as np
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from hyperactive.experiment.integrations import SklearnCvExperiment
from hyperactive.opt.gfo import RandomRestartHillClimbing

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

# Configure Random Restart Hill Climbing
optimizer = RandomRestartHillClimbing(
    search_space=search_space,
    n_iter=40,
    random_state=42,
    experiment=experiment
)

# Run optimization
# Random restart hill climbing performs multiple hill climbing runs
# Each run starts from a different random point in the search space
# The algorithm keeps track of the best solution found across all runs
# This approach combines the efficiency of local search with better global coverage
best_params = optimizer.solve()

# Results
print("\n=== Results ===")
print(f"Best parameters: {best_params}")
print("Random restart hill climbing optimization completed successfully")
