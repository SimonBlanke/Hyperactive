import time
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_boston
from hyperactive import Hyperactive

data = load_boston()
X, y = data.data, data.target


def model(opt):
    gbr = DecisionTreeRegressor(
        max_depth=opt["max_depth"],
        min_samples_split=opt["min_samples_split"],
    )
    scores = cross_val_score(gbr, X, y, cv=10)

    return scores.mean()


search_space = {
    "max_depth": list(range(10, 35)),
    "min_samples_split": list(range(2, 22)),
}

c_time1 = time.time()
hyper = Hyperactive()
hyper.add_search(model, search_space, n_iter=100)
hyper.run()
d_time1 = time.time() - c_time1
print("Optimization time 1:", round(d_time1, 2))

# Hyperactive collects the search data
search_data = hyper.results(model)

# You can pass the search data to memory_warm_start to save time
c_time2 = time.time()
hyper = Hyperactive()
hyper.add_search(model, search_space, n_iter=100, memory_warm_start=search_data)
# The next run will be faster, because Hyperactive knows parts of the search space
hyper.run()
d_time2 = time.time() - c_time2
print("Optimization time 2:", round(d_time2, 2))
