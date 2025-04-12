import numpy as np
from hyperactive import Hyperactive


class Experiment(BaseExperiment):
    def objective_function(self, opt):
        score = -opt["x1"] * opt["x1"]
        return score


experiment = Experiment()

search_config = SearchConfig(
    x1=list(np.arange(-100, 101, 1)),
)


hyper = Hyperactive()
hyper.add_search(experiment, search_config, n_iter=30, memory=True)
hyper.run(verbosity=False)
