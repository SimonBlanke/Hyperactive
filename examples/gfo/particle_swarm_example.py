"""
Particle Swarm Optimization Example - Swarm Intelligence

Particle Swarm Optimization (PSO) is inspired by social behavior of bird flocking
or fish schooling. It maintains a population of candidate solutions (particles)
that move through the search space influenced by their own best position and
the global best position found by the swarm.

Characteristics:
- Population-based metaheuristic
- Inspired by swarm intelligence in nature
- Good balance of exploration and exploitation
- Effective for continuous optimization problems
- Particles share information about promising regions
"""

import numpy as np
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from hyperactive.experiment.integrations import SklearnCvExperiment
from hyperactive.opt.gfo import ParticleSwarmOptimizer

# Load dataset
X, y = load_wine(return_X_y=True)
print(f"Dataset: Wine classification ({X.shape[0]} samples, {X.shape[1]} features)")

# Create experiment
estimator = RandomForestClassifier(random_state=42)
experiment = SklearnCvExperiment(estimator=estimator, X=X, y=y, cv=3)

# Define search space - discrete values for PSO
search_space = {
    "n_estimators": list(range(10, 201, 5)),     # Discrete integer values (step 5)
    "max_depth": list(range(1, 21)),             # Discrete integer values
    "min_samples_split": list(range(2, 21)),     # Discrete integer values
    "min_samples_leaf": list(range(1, 11)),      # Discrete integer values
}

# Configure Particle Swarm Optimization
# The swarm will explore the space collectively, sharing information
# about promising regions through particle communication
optimizer = ParticleSwarmOptimizer(
    search_space=search_space,
    n_iter=50,
    random_state=42,
    experiment=experiment
)

# Run optimization
# PSO maintains a swarm of particles that move through the search space
# Each particle remembers its personal best and is influenced by the global best
# This creates a balance between individual exploration and collective exploitation
best_params = optimizer.solve()

# Results
print("\n=== Results ===")
print(f"Best parameters: {best_params}")
print("Particle swarm optimization completed successfully")
