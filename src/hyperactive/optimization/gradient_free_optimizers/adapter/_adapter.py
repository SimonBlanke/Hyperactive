from ._objective_function import ObjectiveFunction
from ._hyper_gradient_conv import HyperGradientConv
from ._results import Results


class Adapter:
    def __init__(self, search_info):
        self.objective_function = ObjectiveFunction
        self.search_space = HyperGradientConv
        self.results = Results
