from typing import Union
from .optimizers.backend_stuff.run_search import run_search
from .optimizers.backend_stuff.results import Results
from .optimizers.backend_stuff.print_results import PrintResults


class CompositeOptimizer:
    def __init__(self, *optimizers):
        self.optimizers = optimizers

    def run(
        self,
        max_time=None,
        distribution: str = "multiprocessing",
        n_processes: Union[str, int] = "auto",
        verbosity: list = ["progress_bar", "print_results", "print_times"],
    ):
        self.verbosity = verbosity

        self.all_opt_pros = {}
        for optimizer in self.optimizers:
            self.all_opt_pros.update(optimizer.opt_pros)

        for opt in self.all_opt_pros.values():
            opt.max_time = max_time

        self.results_list = run_search(
            self.all_opt_pros, distribution, n_processes
        )

        self.results_ = Results(self.results_list, self.all_opt_pros)

        self._print_info()

    def _print_info(self):
        print_res = PrintResults(self.all_opt_pros, self.verbosity)

        if self.verbosity:
            for _ in range(len(self.all_opt_pros)):
                print("")

        for results in self.results_list:
            nth_process = results["nth_process"]
            print_res.print_process(results, nth_process)
