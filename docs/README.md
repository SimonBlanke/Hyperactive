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

def model(para, X, y):
    model = GradientBoostingClassifier(n_estimators=para['n_estimators'])
    scores = cross_val_score(model, X, y)

    return scores.mean()

search_config = {
    model: {'n_estimators': range(10, 200, 10)}
}

opt = Hyperactive(search_config)
opt.search(X, y)
```
