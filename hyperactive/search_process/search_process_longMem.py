# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from .search_process_shortMem import SearchProcessShortMem


class SearchProcessLongMem(SearchProcessShortMem):
    def __init__(
        self,
        nth_process,
        verb,
        objective_function,
        search_space,
        n_iter,
        function_parameter,
        optimizer,
        n_jobs,
        init_para,
        memory,
        hyperactive,
        random_state,
    ):
        super().__init__(
            nth_process,
            verb,
            objective_function,
            search_space,
            n_iter,
            function_parameter,
            optimizer,
            n_jobs,
            init_para,
            memory,
            hyperactive,
            random_state,
        )

        self._load_memory()

    def _load_memory(self):
        pass

    def store_memory(self, memory):
        pass
