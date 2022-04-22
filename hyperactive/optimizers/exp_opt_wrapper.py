from .gfo_wrapper import _BaseOptimizer_


def try_import_experimental_opt():
    try:
        from experimental_optimization_strategies import (
            RandomAnnealingOptimizer as _RandomAnnealingOptimizer_,
            ParallelAnnealingOptimizer as _ParallelAnnealingOptimizer_,
            EnsembleOptimizer as _EnsembleOptimizer_,
            LocalBayesianOptimizer as _LocalBayesianOptimizer_,
            VariableResolutionBayesianOptimizer as _VariableResolutionBayesianOptimizer_,
            EvoSubSpaceBayesianOptimizer as _EvoSubSpaceBayesianOptimizer_,
        )
    except ImportError as e:
        pass
    else:

        class RandomAnnealingOptimizer(_BaseOptimizer_):
            def __init__(self, **opt_params):
                super().__init__(**opt_params)
                self._OptimizerClass = _RandomAnnealingOptimizer_

        class ParallelAnnealingOptimizer(_BaseOptimizer_):
            def __init__(self, **opt_params):
                super().__init__(**opt_params)
                self._OptimizerClass = _ParallelAnnealingOptimizer_

        class EnsembleOptimizer(_BaseOptimizer_):
            def __init__(self, **opt_params):
                super().__init__(**opt_params)
                self._OptimizerClass = _EnsembleOptimizer_

        class LocalBayesianOptimizer(_BaseOptimizer_):
            def __init__(self, **opt_params):
                super().__init__(**opt_params)
                self._OptimizerClass = _LocalBayesianOptimizer_

        class VariableResolutionBayesianOptimizer(_BaseOptimizer_):
            def __init__(self, **opt_params):
                super().__init__(**opt_params)
                self._OptimizerClass = _VariableResolutionBayesianOptimizer_

        class EvoSubSpaceBayesianOptimizer(_BaseOptimizer_):
            def __init__(self, **opt_params):
                super().__init__(**opt_params)
                self._OptimizerClass = _EvoSubSpaceBayesianOptimizer_
