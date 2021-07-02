from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.datasets import load_boston

from hyperactive import Hyperactive

# import the ProgressBoard
from hyperactive.dashboards import ProgressBoard

data = load_boston()
X, y = data.data, data.target


def model_gbr(opt):
    gbr = GradientBoostingRegressor(
        n_estimators=opt["n_estimators"],
        max_depth=opt["max_depth"],
        min_samples_split=opt["min_samples_split"],
    )
    scores = cross_val_score(gbr, X, y, cv=5)

    return scores.mean()


def model_rfr(opt):
    gbr = RandomForestRegressor(
        n_estimators=opt["n_estimators"],
        min_samples_split=opt["min_samples_split"],
        min_samples_leaf=opt["min_samples_leaf"],
    )
    scores = cross_val_score(gbr, X, y, cv=5)

    return scores.mean()


search_space_gbr = {
    "n_estimators": list(range(30, 200, 5)),
    "max_depth": list(range(2, 12)),
    "min_samples_split": list(range(2, 22)),
}


search_space_rfr = {
    "n_estimators": list(range(10, 100, 1)),
    "min_samples_split": list(range(2, 22)),
    "min_samples_leaf": list(range(2, 22)),
}
# create an instance of the ProgressBoard
progress_board1 = ProgressBoard()


"""
Maybe you do not want to have the information of both searches on the same browser tab?
If you want to open multiple progres board tabs at the same time you can just create 
as many instances of the ProgressBoard-class as you want and pass it two the corresponding 
searches.
"""
# progress_board2 = ProgressBoard()
"""
uncomment the line above and pass progress_board2 
to one .add_search(...) to open two browser tabs at the same time
"""


hyper = Hyperactive()
hyper.add_search(
    model_gbr,
    search_space_gbr,
    n_iter=200,
    n_jobs=2,  # the progress board works seamlessly with multiprocessing
    progress_board=progress_board1,  # pass the instance of the ProgressBoard to .add_search(...)
)
# if you add more searches to Hyperactive they will appear in the same progress board
hyper.add_search(
    model_rfr,
    search_space_rfr,
    n_iter=200,
    n_jobs=4,
    progress_board=progress_board1,
)
# a terminal will open, which opens a dashboard in your browser
hyper.run()
