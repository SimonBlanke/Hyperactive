from typing import Union
from .optimizers.backend_stuff.run_search import run_search
from .optimizers.backend_stuff.results import Results
from .optimizers.backend_stuff.print_results import PrintResults


class CompositeOptimizer:
    optimizers: list

    def __init__(self, *optimizers):
        self.optimizers = list(optimizers)

    def __add__(self, optimizer_instance):
        self.optimizers.append(optimizer_instance)
        return self

    def run(
        self,
        max_time=None,
        distribution: str = "multiprocessing",
        n_processes: Union[str, int] = "auto",
        verbosity: list = ["progress_bar", "print_results", "print_times"],
    ):
        self.verbosity = verbosity

        self.collected_searches = []
        for optimizer in self.optimizers:
            self.collected_searches += optimizer.searches

        for nth_process, search in enumerate(self.collected_searches):
            search.pass_args(max_time, nth_process)

        self.results_list = run_search(
            self.collected_searches, distribution, n_processes
        )

        self.results_ = Results(self.results_list, self.collected_searches)

        self._print_info()

    def _print_info(self):
        print_res = PrintResults(self.collected_searches, self.verbosity)

        if self.verbosity:
            for _ in range(len(self.collected_searches)):
                print("")

        for results in self.results_list:
            nth_process = results["nth_process"]
            print_res.print_process(results, nth_process)
