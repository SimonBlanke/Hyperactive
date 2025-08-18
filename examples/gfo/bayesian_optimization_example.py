"""
Bayesian Optimization Example - Gaussian Process-based Optimization

Bayesian optimization uses a probabilistic model (typically Gaussian Process) to
model the objective function and an acquisition function to decide where to sample
next. This approach is highly sample-efficient and particularly useful when 
function evaluations are expensive.

Characteristics:
- Sample-efficient optimization
- Uses Gaussian Process to model objective function
- Balances exploration vs exploitation via acquisition functions
- Excellent for expensive function evaluations
- Handles uncertainty quantification naturally
"""

import numpy as np
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from hyperactive.experiment.integrations import SklearnCvExperiment
from hyperactive.opt.gfo import BayesianOptimizer

# Load dataset  
X, y = load_wine(return_X_y=True)
print(f"Dataset: Wine classification ({X.shape[0]} samples, {X.shape[1]} features)")

# Create experiment
estimator = RandomForestClassifier(random_state=42)
experiment = SklearnCvExperiment(estimator=estimator, X=X, y=y, cv=3)

# Define search space - discrete values for Bayesian optimization
search_space = {
    "n_estimators": list(range(10, 201, 10)),    # Discrete integer values (step 10)
    "max_depth": list(range(1, 21)),             # Discrete integer values
    "min_samples_split": list(range(2, 21)),     # Discrete integer values
    "min_samples_leaf": list(range(1, 11)),      # Discrete integer values
}

# Configure Bayesian Optimization
# Provide some initial good points to help the GP model initialization
warm_start_points = [
    {"n_estimators": 100, "max_depth": 10, "min_samples_split": 5, "min_samples_leaf": 2},
    {"n_estimators": 50, "max_depth": 15, "min_samples_split": 3, "min_samples_leaf": 1}
]

optimizer = BayesianOptimizer(
    search_space=search_space,
    n_iter=35,
    random_state=42,
    initialize={"warm_start": warm_start_points},
    experiment=experiment
)

# Run optimization
# Bayesian optimization builds a GP model of the objective function
# and uses acquisition functions (like Expected Improvement) to select
# the most promising points to evaluate next
best_params = optimizer.solve()

# Results
print("\n=== Results ===")  
print(f"Best parameters: {best_params}")
print("Bayesian optimization completed successfully")