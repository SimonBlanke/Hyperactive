import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from hyperactive import Hyperactive

data = load_breast_cancer()
X, y = data.data, data.target


def meta_opt(opt):
    scores = []

    for i in range(100):

        def sphere_function(opt):
            loss = []
            for key in opt.keys():
                loss.append(opt[key] * opt[key])

            return -np.array(loss).sum()

        dim_size = list(np.arange(-10, 10, 0.01))

        search_space = {
            "x1": dim_size,
            "x2": dim_size,
        }

        hyper = Hyperactive(verbosity=0)
        hyper.add_search(
            sphere_function,
            search_space,
            n_iter=100,
            optimizer={
                "EvolutionStrategy": {
                    "individuals": opt["individuals"],
                    "mutation_rate": opt["mutation_rate"],
                }
            },
            random_state=i,
        )
        hyper.run()

        scores.append(hyper.score_best["sphere_function.0"])

    return np.array(scores).sum()


search_space = {
    "individuals": list(range(2, 21)),
    "mutation_rate": list(np.arange(0, 1, 0.1)),
}


hyper = Hyperactive()
hyper.add_search(meta_opt, search_space, n_iter=300, optimizer="RandomSearch")
hyper.run()
