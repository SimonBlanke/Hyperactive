## Sklearn

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


opt = Hyperactive(X, y)
opt.search(search_config, n_iter=100)
```

## XGBoost

```python
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from sklearn.datasets import load_breast_cancer
from hyperactive import Hyperactive

data = load_breast_cancer()
X, y = data.data, data.target


def model(para, X, y):
    xgb = XGBClassifier(
        n_estimators=para["n_estimators"],
        max_depth=para["max_depth"],
        learning_rate=para["learning_rate"],
    )
    scores = cross_val_score(xgb, X, y, cv=3)

    return scores.mean()


search_config = {
    model: {
        "n_estimators": range(10, 200, 10),
        "max_depth": range(2, 12),
        "learning_rate": [1e-3, 1e-2, 1e-1, 0.5, 1.0],
    }
}


opt = Hyperactive(X, y)
opt.search(search_config, n_iter=100)
```

## LightGBM

```python
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMRegressor
from sklearn.datasets import load_breast_cancer
from hyperactive import Hyperactive

data = load_breast_cancer()
X, y = data.data, data.target


def model(para, X, y):
    lgbm = LGBMRegressor(
        num_leaves=para["num_leaves"],
        bagging_freq=para["bagging_freq"],
        learning_rate=para["learning_rate"],
    )
    scores = cross_val_score(lgbm, X, y, cv=3)

    return scores.mean()


search_config = {
    model: {
        "num_leaves": range(2, 20),
        "bagging_freq": range(2, 12),
        "learning_rate": [1e-3, 1e-2, 1e-1, 0.5, 1.0],
    }
}


opt = Hyperactive(X, y)
opt.search(search_config, n_iter=30)
```

## CatBoost

```python
from sklearn.model_selection import cross_val_score
from catboost import CatBoostClassifier
from sklearn.datasets import load_breast_cancer
from hyperactive import Hyperactive

data = load_breast_cancer()
X, y = data.data, data.target


def model(para, X, y):
    cbc = CatBoostClassifier(
        iterations=10, depth=para["depth"], learning_rate=para["learning_rate"]
    )
    scores = cross_val_score(cbc, X, y, cv=3)

    return scores.mean()


search_config = {
    model: {"depth": range(2, 22), "learning_rate": [1e-3, 1e-2, 1e-1, 0.5, 1.0]}
}


opt = Hyperactive(X, y)
opt.search(search_config, n_iter=5)
```

## RGF

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from rgf.sklearn import RGFClassifier

from hyperactive import Hyperactive

data = load_breast_cancer()
X, y = data.data, data.target


def model(para, X, y):
    rgf = RGFClassifier(
        max_leaf=para["max_leaf"],
        reg_depth=para["reg_depth"],
        min_samples_leaf=para["min_samples_leaf"],
        algorithm="RGF_Sib",
        test_interval=100,
        verbose=False,
    )
    scores = cross_val_score(rgf, X, y, cv=3)

    return scores.mean()


search_config = {
    model: {
        "max_leaf": range(1000, 10000, 100),
        "reg_depth": range(1, 21),
        "min_samples_leaf": range(1, 21),
    }
}

opt = Hyperactive(X, y)
opt.search(search_config, n_iter=5)
```

## Mlxtend

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from mlxtend.classifier import EnsembleVoteClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from hyperactive import Hyperactive

data = load_breast_cancer()
X, y = data.data, data.target


def model(para, X, y):
    gbc = GradientBoostingClassifier(
        n_estimators=para["n_estimators"], max_depth=para["max_depth"]
    )
    mlp = MLPClassifier(hidden_layer_sizes=para["hidden_layer_sizes"])
    svc = SVC(gamma="auto", probability=True)

    eclf = EnsembleVoteClassifier(
        clfs=[gbc, mlp, svc], weights=[2, 1, 1], voting="soft"
    )

    scores = cross_val_score(eclf, X, y, cv=3)

    return scores.mean()


search_config = {
    model: {
        "n_estimators": range(10, 100, 10),
        "max_depth": range(2, 12),
        "hidden_layer_sizes": (range(10, 100, 10),),
    }
}


opt = Hyperactive(X, y)
opt.search(search_config, n_iter=30)
```

