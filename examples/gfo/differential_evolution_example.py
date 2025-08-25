"""
Differential Evolution Example - Vector-Based Evolutionary Algorithm

Differential Evolution is a population-based evolutionary algorithm that uses
difference vectors between population members to generate new candidate solutions.
It's particularly effective for continuous optimization problems and has robust
convergence properties across various problem types.

Characteristics:
- Population-based with vector difference operations
- Self-adapting through mutation and crossover
- Robust convergence across different problem types
- Effective for continuous and mixed optimization
- Simple parameter control with good default behavior
"""

import numpy as np
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from hyperactive.experiment.integrations import SklearnCvExperiment
from hyperactive.opt.gfo import DifferentialEvolution

# Load dataset
X, y = load_wine(return_X_y=True)
print(f"Dataset: Wine classification ({X.shape[0]} samples, {X.shape[1]} features)")

# Create experiment
estimator = RandomForestClassifier(random_state=42)
experiment = SklearnCvExperiment(estimator=estimator, X=X, y=y, cv=3)

# Define search space
search_space = {
    "n_estimators": list(range(10, 201, 5)),     # Discrete integer values (broader range)
    "max_depth": list(range(1, 21)),             # Discrete integer values
    "min_samples_split": list(range(2, 21)),     # Discrete integer values
    "min_samples_leaf": list(range(1, 11)),      # Discrete integer values
}

# Configure Differential Evolution
optimizer = DifferentialEvolution(
    search_space=search_space,
    n_iter=50,
    random_state=42,
    experiment=experiment
)

# Run optimization
# Differential evolution uses difference vectors between population members
# For each target vector, it creates a mutant vector using weighted differences
# from other population members, then applies crossover to generate trial vector
# The trial vector replaces the target if it has better fitness
best_params = optimizer.solve()

# Results
print("\n=== Results ===")
print(f"Best parameters: {best_params}")
print("Differential evolution optimization completed successfully")
