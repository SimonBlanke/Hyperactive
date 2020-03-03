## Introduction

Hyperactive is primarly a <b>hyperparameter optimization toolkit</b>, that aims to simplify the model-selection and -tuning process. You can use any machine- or deep-learning package and it is not necessary to learn new syntax. Hyperactive offers <b>high versatility</b> in model optimization because of two characteristics:

  - You can define any kind of model in the objective function. It just has to return a score/metric that gets maximized.
  - The search space accepts not just int, float or str as data types but even functions, classes or any python objects.


## Installation

?> PyPi always has the newest version of hyperactive:

[![pyversions](https://img.shields.io/pypi/pyversions/hyperactive.svg?style=for-the-badge&logo=python&color=blue&logoColor=white)](https://pypi.org/project/hyperactive)
[![PyPI version](https://img.shields.io/pypi/v/hyperactive?style=for-the-badge&logo=pypi&color=green&logoColor=white)](https://pypi.org/project/hyperactive/)
[![PyPI version](https://img.shields.io/pypi/dm/hyperactive?style=for-the-badge&color=red)](https://pypi.org/project/hyperactive/)

```bash
pip install hyperactive
```

## Minimal example

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from hyperactive import Hyperactive

data = load_breast_cancer()
X, y = data.data, data.target

'''define the model in a function'''
def model(para, X, y):
    '''optimize one or multiple hyperparameters'''
    gbc = GradientBoostingClassifier(n_estimators=para['n_estimators'])
    scores = cross_val_score(gbc, X, y)

    return scores.mean()

'''create the search space and search_config'''
search_config = {
    model: {'n_estimators': range(10, 200, 10)}
}

'''start the optimization run'''
opt = Hyperactive(X, y)
opt.search(search_config, n_iter=20)
```
