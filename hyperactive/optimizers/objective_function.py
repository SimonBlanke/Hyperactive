# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from .dictionary import DictClass


def gfo2hyper(search_space, para):
    values_dict = {}
    for i, key in enumerate(search_space.keys()):
        pos_ = int(para[key])
        values_dict[key] = search_space[key][pos_]

    return values_dict


class ObjectiveFunction(DictClass):
    def __init__(self, objective_function, optimizer):
        super().__init__()

        self.objective_function = objective_function
        self.optimizer = optimizer

    def __call__(self, search_space, data_c):
        # wrapper for GFOs
        def _model(para):
            para = gfo2hyper(search_space, para)
            self.para_dict = para
            results = self.objective_function(self)

            if data_c:
                progress_dict = para

                if isinstance(results, tuple):
                    score = results[0]
                    results_dict = results[1]
                else:
                    score = results
                    results_dict = {}

                results_dict["score"] = score

                progress_dict.update(results_dict)
                progress_dict["score_best"] = self._optimizer.score_best
                progress_dict["nth_iter"] = self._optimizer.nth_iter

                data_c.save_iter(progress_dict)

            # ltm save after iteration
            # self.ltm.ltm_obj_func_wrapper(results, para, nth_process)

            return results

        _model.__name__ = self.objective_function.__name__
        return _model
