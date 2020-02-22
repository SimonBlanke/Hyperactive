## Memory

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_iris
from hyperactive import Hyperactive

iris_data = load_iris()
X, y = iris_data.data, iris_data.target


def model(para, X, y):
    gbc = GradientBoostingClassifier(
        n_estimators=para["n_estimators"], max_depth=para["max_depth"]
    )
    scores = cross_val_score(gbc, X, y, cv=3)

    return scores.mean()


search_config = {model: {"n_estimators": range(10, 200, 10), "max_depth": range(2, 15)}}

"""
The memory will remember previous evaluations done during the optimization process.
Instead of retraining the model, it accesses the memory and uses the saved score/loss.
This shows as a speed up during the optimization process, since the whole search space has been explored.
"""
opt = Hyperactive(X, y, memory="short")
opt.search(search_config, n_iter=1000)
```

## Scatter initialization

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from hyperactive import Hyperactive

data = load_breast_cancer()
X, y = data.data, data.target


def model(para, X, y):
    gbc = GradientBoostingClassifier(
        n_estimators=para["n_estimators"],
        max_depth=para["max_depth"],
        min_samples_split=para["min_samples_split"],
    )
    scores = cross_val_score(gbc, X, y, cv=3)

    return scores.mean()


search_config = {
    model: {
        "n_estimators": range(10, 200, 10),
        "max_depth": range(2, 12),
        "min_samples_split": range(2, 12),
    }
}

# Without scatter initialization
opt = Hyperactive(X, y)
opt.search(
    search_config,
    optimizer="HillClimbing",
    n_iter=10,
    random_state=0,
    init_config=False,
)

init_config = {
    model: {"scatter_init": 10}
}

# With scatter initialization
opt = Hyperactive(X, y)
opt.search(
    search_config,
    optimizer="HillClimbing",
    n_iter=10,
    random_state=0,
    init_config=init_config,
)
```

## Warm start

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from hyperactive import Hyperactive

data = load_breast_cancer()
X, y = data.data, data.target


def model(para, X, y):
    gbc = GradientBoostingClassifier(
        n_estimators=para["n_estimators"],
        max_depth=para["max_depth"],
        min_samples_split=para["min_samples_split"],
    )
    scores = cross_val_score(gbc, X, y, cv=3)

    return scores.mean()


search_config = {
    model: {
        "n_estimators": range(10, 200, 10),
        "max_depth": range(2, 12),
        "min_samples_split": range(2, 12),
    }
}

init_config = {
    model: {"n_estimators": [190], "max_depth": [2], "min_samples_split": [5]}
}


opt = Hyperactive(search_config, init_config=init_config)
opt.search(X, y)
```

