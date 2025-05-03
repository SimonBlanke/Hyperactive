# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import logging
import numpy as np

indent = "  "


class PrintResults:
    def __init__(self, opt_pros, verbosity):
        self.opt_pros = opt_pros
        self.verbosity = verbosity

    def _print_times(self, eval_time, iter_time, n_iter):
        opt_time = iter_time - eval_time
        iterPerSec = n_iter / iter_time

        print(
            indent,
            "Evaluation time   :",
            eval_time,
            "sec",
            indent,
            "[{} %]".format(round(eval_time / iter_time * 100, 2)),
        )
        print(
            indent,
            "Optimization time :",
            opt_time,
            "sec",
            indent,
            "[{} %]".format(round(opt_time / iter_time * 100, 2)),
        )
        if iterPerSec >= 1:
            print(
                indent,
                "Iteration time    :",
                iter_time,
                "sec",
                indent,
                "[{} iter/sec]".format(round(iterPerSec, 2)),
            )
        else:
            secPerIter = iter_time / n_iter
            print(
                indent,
                "Iteration time    :",
                iter_time,
                "sec",
                indent,
                "[{} sec/iter]".format(round(secPerIter, 2)),
            )
        print(" ")

    def align_para_names(self, para_names):
        str_lengths = [len(str_) for str_ in para_names]
        max_length = max(str_lengths)

        para_names_align = {}
        for para_name, str_length in zip(para_names, str_lengths):
            added_spaces = max_length - str_length
            para_names_align[para_name] = " " * added_spaces

        return para_names_align

    def _print_results(
        self,
        objective_function,
        best_score,
        best_para,
        best_iter,
        best_additional_results,
        random_seed,
    ):
        print("\nResults: '{}'".format(objective_function.__name__), " ")
        if best_para is None:
            print(indent, "Best score:", best_score, " ")
            print(indent, "Best parameter set:", best_para, " ")
            print(indent, "Best iteration:", best_iter, " ")

        else:
            print(indent, "Best score:", best_score, " ")

            if best_additional_results:
                print(indent, "Best additional results:")
                add_results_names = list(best_additional_results.keys())
                add_results_names_align = self.align_para_names(add_results_names)

                for best_additional_result in best_additional_results.keys():
                    added_spaces = add_results_names_align[best_additional_result]
                    print(
                        indent,
                        indent,
                        "'{}'".format(best_additional_result),
                        "{}:".format(added_spaces),
                        best_additional_results[best_additional_result],
                        " ",
                    )

            if best_para:
                print(indent, "Best parameter set:")
                para_names = list(best_para.keys())
                para_names_align = self.align_para_names(para_names)

                for para_key in best_para.keys():
                    added_spaces = para_names_align[para_key]
                    print(
                        indent,
                        indent,
                        "'{}'".format(para_key),
                        "{}:".format(added_spaces),
                        best_para[para_key],
                        " ",
                    )

            print(indent, "Best iteration:", best_iter, " ")

        print(" ")
        print(indent, "Random seed:", random_seed, " ")
        print(" ")

    def print_process(self, results, nth_process):
        verbosity = self.verbosity
        objective_function = self.opt_pros[nth_process].objective_function
        search_space = self.opt_pros[nth_process].s_space.search_space

        search_data = results["search_data"]

        try:
            best_sample = search_data.iloc[search_data["score"].idxmax()]

        except TypeError:
            logging.warning(
                "Warning: Cannot index by location index with a non-integer key"
            )

        else:
            best_score = best_sample["score"]
            best_values = best_sample[list(search_space.keys())]
            best_para = dict(zip(list(search_space.keys()), best_values))
            best_additional_results_df = best_sample.drop(
                ["score"] + list(search_space.keys())
            )
            best_additional_results = best_additional_results_df.to_dict()

            best_iter = results["best_iter"]
            eval_times = results["eval_times"]
            iter_times = results["iter_times"]
            random_seed = results["random_seed"]

            n_iter = self.opt_pros[nth_process].n_iter

            eval_time = np.array(eval_times).sum()
            iter_time = np.array(iter_times).sum()

            if "print_results" in verbosity:
                self._print_results(
                    objective_function,
                    best_score,
                    best_para,
                    best_iter,
                    best_additional_results,
                    random_seed,
                )

            if "print_times" in verbosity:
                self._print_times(eval_time, iter_time, n_iter)
