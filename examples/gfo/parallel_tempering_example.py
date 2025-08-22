"""
Parallel Tempering Example - Multi-Temperature Markov Chain Monte Carlo

Parallel Tempering runs multiple Markov chains at different temperatures
simultaneously, allowing high-temperature chains to explore globally while
low-temperature chains exploit locally. Chains can exchange information,
combining global exploration with local refinement in a principled way.

Characteristics:
- Multiple parallel chains at different temperatures
- Information exchange between chains (replica exchange)
- High-temperature chains provide global exploration
- Low-temperature chains provide local exploitation
- Effective for complex multi-modal optimization landscapes
"""

import numpy as np
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from hyperactive.experiment.integrations import SklearnCvExperiment
from hyperactive.opt.gfo import ParallelTempering

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

# Configure Parallel Tempering
optimizer = ParallelTempering(
    search_space=search_space,
    n_iter=45,
    random_state=42,
    experiment=experiment
)

# Run optimization
# Parallel tempering maintains multiple chains at different temperatures
# High-temperature chains can easily accept worse moves (exploration)
# Low-temperature chains focus on local improvements (exploitation)
# Chains periodically attempt to exchange states based on temperature differences
# This provides both global search capability and local refinement
best_params = optimizer.solve()

# Results
print("\n=== Results ===")
print(f"Best parameters: {best_params}")
print("Parallel tempering optimization completed successfully")
