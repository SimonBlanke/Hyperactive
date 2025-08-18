"""
DIRECT Algorithm Example - Dividing Rectangles Global Optimization

DIRECT (DIviding RECTangles) is a global optimization algorithm that systematically
divides the search space into rectangles and evaluates their potential for containing
the global optimum. It balances global search with local refinement by focusing
computational effort on the most promising regions.

Characteristics:
- Global optimization through systematic space division
- Balances exploration of large regions with exploitation of promising areas
- Lipschitz constant estimation for convergence guarantees
- No user-specified parameters required
- Effective for global optimization of continuous functions
"""

import numpy as np
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from hyperactive.experiment.integrations import SklearnCvExperiment
from hyperactive.opt.gfo import DirectAlgorithm

# Load dataset
X, y = load_wine(return_X_y=True)
print(f"Dataset: Wine classification ({X.shape[0]} samples, {X.shape[1]} features)")

# Create experiment
estimator = RandomForestClassifier(random_state=42)
experiment = SklearnCvExperiment(estimator=estimator, X=X, y=y, cv=3)

# Define search space - DIRECT works well with moderate-sized discrete spaces
search_space = {
    "n_estimators": list(range(10, 201, 20)),    # Discrete integer values (reduced)
    "max_depth": list(range(1, 21, 2)),          # Discrete integer values (reduced)
    "min_samples_split": list(range(2, 21, 2)),  # Discrete integer values (reduced)
    "min_samples_leaf": list(range(1, 11)),      # Discrete integer values
}

# Configure DIRECT Algorithm
optimizer = DirectAlgorithm(
    search_space=search_space,
    n_iter=30,
    random_state=42,
    experiment=experiment
)

# Run optimization
# DIRECT divides the search space into hyperrectangles and maintains
# a list of potentially optimal rectangles. It systematically subdivides
# the most promising rectangles, balancing global exploration with
# local exploitation based on function values and rectangle sizes
best_params = optimizer.solve()

# Results
print("\n=== Results ===")
print(f"Best parameters: {best_params}")
print("DIRECT algorithm optimization completed successfully")