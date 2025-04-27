from typing import Union
from ._distribution import run_search
from ._results import Results
from ._print_results import PrintResults

from ._run_info import RunInfo


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
        if not verbosity:
            verbosity = []

        run_info = RunInfo(
            max_time,
            distribution,
            n_processes,
            verbosity,
        )

        self.collected_searches = []
        for optimizer in self.optimizers:
            self.collected_searches += optimizer.searches

        for nth_process, search in enumerate(self.collected_searches):
            search.pass_args(max_time, nth_process, verbosity)

        self.results_list = run_search(
            self.collected_searches, distribution, n_processes
        )

        self.results_ = Results(self.results_list, self.collected_searches)

        self._print_info(verbosity)

    def _print_info(self, verbosity):
        print_res = PrintResults(self.collected_searches, verbosity)

        if verbosity:
            for _ in range(len(self.collected_searches)):
                print("")

        for results in self.results_list:
            nth_process = results["nth_process"]
            print_res.print_process(results, nth_process)
