import time
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import load_wine
from hyperactive import Hyperactive

data = load_wine()
X, y = data.data, data.target

"""
Hyperactive cannot handle multi objective optimization. 
But we can achive something similar with a workaround.
The following example searches for the highest cv-score and the lowest training time.
It is possible by creating an objective/score from those two variables.
You can also return additional parameters to track the cv-score and training time separately.
"""


def model(opt):
    gbr = GradientBoostingRegressor(
        n_estimators=opt["n_estimators"],
        max_depth=opt["max_depth"],
        min_samples_split=opt["min_samples_split"],
    )

    c_time = time.time()
    scores = cross_val_score(gbr, X, y, cv=3)
    train_time = time.time() - c_time

    cv_score = scores.mean()

    # you can create a score that is a composition of two objectives
    score = cv_score / train_time

    # instead of just returning the score you can also return the score + a dict
    return score, {"training_time": train_time, "cv_score": cv_score}


search_space = {
    "n_estimators": list(range(10, 150, 5)),
    "max_depth": list(range(2, 12)),
    "min_samples_split": list(range(2, 22)),
}


hyper = Hyperactive()
hyper.add_search(model, search_space, n_iter=20)
hyper.run()

# The variables from the dict are collected in the results.
print("\n Results \n", hyper.search_data(model))
