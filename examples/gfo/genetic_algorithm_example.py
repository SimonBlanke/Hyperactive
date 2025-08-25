"""
Genetic Algorithm Example - Evolutionary Computation

Genetic Algorithm is inspired by biological evolution, maintaining a population
of candidate solutions that evolve over generations through selection, crossover,
and mutation operations. It explores multiple regions simultaneously and can
handle complex, multi-modal optimization landscapes effectively.

Characteristics:
- Population-based evolutionary approach
- Uses crossover (recombination) and mutation operators
- Selection pressure drives evolution toward better solutions
- Good for complex, multi-modal optimization problems
- Naturally parallel and robust to noise
"""

import numpy as np
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from hyperactive.experiment.integrations import SklearnCvExperiment
from hyperactive.opt.gfo import GeneticAlgorithm

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

# Configure Genetic Algorithm
# Population-based approach works well without specific warm start points
optimizer = GeneticAlgorithm(
    search_space=search_space,
    n_iter=50,
    random_state=42,
    experiment=experiment
)

# Run optimization
# Genetic algorithm maintains a population of candidate solutions
# Each generation applies selection, crossover, and mutation operations
# Fitter individuals have higher probability of reproduction
# Over time, the population evolves toward better solutions
best_params = optimizer.solve()

# Results
print("\n=== Results ===")
print(f"Best parameters: {best_params}")
print("Genetic algorithm optimization completed successfully")
