from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import load_boston

from hyperactive import Hyperactive

# import the ProgressBoard
from hyperactive.dashboards import ProgressBoard

data = load_boston()
X, y = data.data, data.target


def model(opt):
    gbr = GradientBoostingRegressor(
        n_estimators=opt["n_estimators"],
        max_depth=opt["max_depth"],
        min_samples_split=opt["min_samples_split"],
    )
    scores = cross_val_score(gbr, X, y, cv=3)

    return scores.mean()


search_space = {
    "n_estimators": list(range(50, 150, 5)),
    "max_depth": list(range(2, 12)),
    "min_samples_split": list(range(2, 22)),
}

# create an instance of the ProgressBoard
progress_board = ProgressBoard()


hyper = Hyperactive()

hyper.add_search(
    model,
    search_space,
    n_iter=120,
    n_jobs=2,  # the progress board works seamlessly with multiprocessing
    progress_board=progress_board,  # pass the instance of the ProgressBoard to .add_search(...)
)

# a terminal will open, which opens a dashboard in your browser
hyper.run()
