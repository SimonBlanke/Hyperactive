from .hyper_optimizer import HyperOptimizer

try:
    from experimental_optimization_algorithms import (
        RandomAnnealingOptimizer as _RandomAnnealingOptimizer_,
    )
except ImportError as e:
    print(e)
else:

    class RandomAnnealingOptimizer(HyperOptimizer):
        def __init__(self, **opt_params):
            super().__init__(**opt_params)
            self._OptimizerClass = _RandomAnnealingOptimizer_
