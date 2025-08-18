"""
Powell's Method Example - Coordinate Descent Optimization

Powell's Method is a derivative-free optimization algorithm that performs
sequential line searches along conjugate directions. It's particularly effective
for smooth, continuous functions and can achieve quadratic convergence on
quadratic functions without requiring gradient information.

Characteristics:
- Sequential line searches along conjugate directions
- Derivative-free optimization method
- Good convergence properties for smooth functions
- Builds conjugate directions automatically
- Effective for continuous optimization problems
"""

import numpy as np
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from hyperactive.experiment.integrations import SklearnCvExperiment
from hyperactive.opt.gfo import PowellsMethod

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

# Configure Powell's Method
warm_start_points = [
    {"n_estimators": 100, "max_depth": 10, "min_samples_split": 5, "min_samples_leaf": 2}
]

optimizer = PowellsMethod(
    search_space=search_space,
    n_iter=35,
    random_state=42,
    initialize={"warm_start": warm_start_points},
    experiment=experiment
)

# Run optimization
# Powell's method performs line searches along coordinate directions initially,
# then constructs conjugate directions from successful search directions
# Each iteration consists of n line searches (where n is the dimension)
# followed by an additional line search along the overall displacement vector
best_params = optimizer.solve()

# Results
print("\n=== Results ===")
print(f"Best parameters: {best_params}")
print("Powell's method optimization completed successfully")
