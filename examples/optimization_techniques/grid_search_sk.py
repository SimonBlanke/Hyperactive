"""
GridSearchSk Optimizer Example

This example demonstrates how to use the GridSearchSk optimizer with 
loky backend for parallel execution.
"""

import numpy as np
from hyperactive.opt import GridSearchSk
from hyperactive.experiment.toy import Sphere

# Define the optimization problem using a toy sphere function
# The sphere function f(x,y) = x² + y² has its minimum at (0,0)
sphere_experiment = Sphere(n_dim=2)

# Define parameter grid - creates a 9x9 = 81 point grid
param_grid = {
    "x0": np.linspace(-2, 2, 9),
    "x1": np.linspace(-2, 2, 9),
}

# Grid search with parallel execution using loky backend
grid_search = GridSearchSk(
    param_grid=param_grid,
    backend="loky",  # Use loky backend for parallelization
    backend_params={
        "n_jobs": -1,  # Use all available CPU cores
    },
    experiment=sphere_experiment,
)

best_params = grid_search.run()
print(f"Best params: {best_params}, Score: {grid_search.best_score_:.6f}")