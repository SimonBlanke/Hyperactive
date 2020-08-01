# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from optimization_metadata import HyperactiveWrapper
from .meta_data.meta_data_path import meta_data_path


class ResultsManagerBase:
    def __init__(
        self,
        search_name,
        objective_function,
        search_space,
        function_parameter,
        verbosity,
    ):
        self.search_name = search_name
        self.objective_function = objective_function
        self.search_space = search_space
        self.function_parameter = function_parameter

        self.memory_dict_new = {}


class ResultsManager(ResultsManagerBase):
    def __init__(
        self,
        search_name,
        objective_function,
        search_space,
        function_parameter,
        verbosity,
    ):
        super().__init__(
            search_name, objective_function, search_space, function_parameter, verbosity
        )


class ResultsManagerMemory(ResultsManagerBase):
    def __init__(
        self,
        search_name,
        objective_function,
        search_space,
        function_parameter,
        verbosity,
    ):
        super().__init__(
            search_name, objective_function, search_space, function_parameter, verbosity
        )

        self.hypermem = HyperactiveWrapper(
            main_path=meta_data_path(),
            X=function_parameter["features"],
            y=function_parameter["target"],
            model=self.objective_function,
            search_space=search_space,
            verbosity=verbosity,
        )

    def load_long_term_memory(self):
        return self.hypermem.load()

    def save_long_term_memory(self):
        self.hypermem.save(self.memory_dict_new)
