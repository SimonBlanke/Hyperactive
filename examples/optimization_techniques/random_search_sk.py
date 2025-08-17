"""
RandomSearchSk Optimizer Example

This example demonstrates how to use the RandomSearchSk optimizer with
sklearn integration and threading backend for parallel execution.
"""

import numpy as np
from scipy.stats import uniform, randint
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from hyperactive.opt import RandomSearchSk
from hyperactive.experiment.integrations import SklearnCvExperiment

# Load iris dataset and create sklearn experiment
X, y = load_iris(return_X_y=True)
sklearn_experiment = SklearnCvExperiment(
    estimator=SVC(),
    X=X,
    y=y,
)

# Define parameter distributions for random search
# Mix of discrete and continuous distributions
param_distributions = {
    "C": uniform(loc=0.1, scale=10),  # Continuous uniform distribution
    "gamma": ["scale", "auto"] + list(np.logspace(-4, 1, 20)),  # Mixed discrete/continuous
    "kernel": ["rbf", "poly", "sigmoid"],  # Discrete choices
}

# Random search with threading backend
random_search = RandomSearchSk(
    param_distributions=param_distributions,
    n_iter=30,  # Number of random samples
    random_state=42,  # For reproducible results
    backend="threading",  # Use threading backend for parallelization
    backend_params={
        "n_jobs": 2,  # Use 2 threads
    },
    experiment=sklearn_experiment,
)

best_params = random_search.run()
print(f"Best params: {best_params}, Score: {random_search.best_score_:.4f}")