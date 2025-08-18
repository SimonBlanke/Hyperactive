"""
Evolution Strategy Example - Self-Adaptive Evolutionary Algorithm

Evolution Strategy is a class of evolutionary algorithms that evolves both the
solution candidates and their associated strategy parameters (like mutation
strengths). This self-adaptive mechanism makes it particularly effective for
continuous optimization without requiring manual parameter tuning.

Characteristics:
- Self-adaptive mutation parameters
- Emphasis on mutation over crossover
- Excellent for continuous optimization problems
- Automatic adaptation to problem landscape
- Robust performance across different problem types
"""

import numpy as np
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from hyperactive.experiment.integrations import SklearnCvExperiment
from hyperactive.opt.gfo import EvolutionStrategy

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

# Configure Evolution Strategy
optimizer = EvolutionStrategy(
    search_space=search_space,
    n_iter=45,
    random_state=42,
    experiment=experiment
)

# Run optimization
# Evolution strategy maintains a population where both solutions and strategy
# parameters (mutation strengths) evolve together. This self-adaptation allows
# the algorithm to automatically adjust its search behavior to match the
# problem characteristics without manual parameter tuning
best_params = optimizer.solve()

# Results
print("\n=== Results ===")
print(f"Best parameters: {best_params}")
print("Evolution strategy optimization completed successfully")
