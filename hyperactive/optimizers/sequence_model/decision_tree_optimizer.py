# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from .bayesian_optimization import BayesianOptimizer


class DecisionTreeOptimizer(BayesianOptimizer):
    """Based on the forest-optimizer in the scikit-optimize package"""

    def __init__(self, _opt_args_):
        super().__init__(_opt_args_)
        self.n_positioners = 1
        self.regr = _opt_args_.tree_regressor
