# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from .general_optimizer import Optimizer


class Hyperactive:
    def __init__(
        self, X, y, random_state=None, verbosity=3, warnings=False, ext_warnings=False,
    ):
        self.X = X
        self.y = y
        self.opt = Optimizer(
            random_state=random_state,
            verbosity=verbosity,
            warnings=warnings,
            ext_warnings=ext_warnings,
            hyperactive=True,
        )

    def add_search(self, *args, **kwargs):
        kwargs["function_parameter"] = {"features": self.X, "target": self.y}

        self.opt.add_search(*args, **kwargs)

    def run(self, max_time=None):
        self.opt.run(max_time=max_time)

