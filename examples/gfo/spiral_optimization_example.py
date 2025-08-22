"""
Spiral Optimization Example - Nature-Inspired Spiral Search

Spiral Optimization is inspired by natural phenomena like spiral galaxies and
logarithmic spirals found in nature. It uses spiral-shaped search patterns
to systematically explore the search space, combining global exploration
with local exploitation through adaptive spiral parameters.

Characteristics:
- Nature-inspired spiral search patterns
- Adaptive spiral parameters for exploration/exploitation balance
- Systematic coverage of search space through spiral trajectories
- Good for continuous and mixed optimization problems
- Unique geometric approach to search space exploration
"""

import numpy as np
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from hyperactive.experiment.integrations import SklearnCvExperiment
from hyperactive.opt.gfo import SpiralOptimization

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

# Configure Spiral Optimization
warm_start_points = [
    {"n_estimators": 100, "max_depth": 10, "min_samples_split": 5, "min_samples_leaf": 3}
]

optimizer = SpiralOptimization(
    search_space=search_space,
    n_iter=40,
    random_state=42,
    initialize={"warm_start": warm_start_points},
    experiment=experiment
)

# Run optimization
# Spiral optimization uses logarithmic or Archimedean spiral patterns
# to explore the search space. The spiral parameters (radius, angle, center)
# adapt based on search progress, allowing both global exploration through
# wide spirals and local exploitation through tight spirals around promising regions
best_params = optimizer.solve()

# Results
print("\n=== Results ===")
print(f"Best parameters: {best_params}")
print("Spiral optimization completed successfully")
