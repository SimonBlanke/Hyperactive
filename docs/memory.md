## Introduction

The memory-module is a part of Hyperactive that remembers the parameters and scores of previous evaluations.

You can choose from three options:

  - Set memory = False to not use memory at all
  - Set memory = "short"
  - Set memory = "long"


## Short term memory

Hyperactive will remember evaluations from the current run. This means, that if the optimizer encounters a positions that has been evaluated before it will look up the score in the memory instead of evaluating it again (which is very time consuming).

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_iris
from hyperactive import Hyperactive

iris_data = load_iris()
X, y = iris_data.data, iris_data.target

def model(para, X, y):
    gbc = GradientBoostingClassifier(n_estimators=para["n_estimators"])
    return cross_val_score(gbc, X, y, cv=3).mean()

search_config = {model: {"n_estimators": range(5, 100)}}

'''
Short term memory speeds up the optimization run every time it encounters a known position.
If it has discovered all positions available in the search space it will quickly run through the optimization.
'''
opt = Hyperactive(X, y, memory="short")
opt.search(search_config, n_iter=500)
```

<br>

## Long term memory

The long term memory saves the parameters and scores that have been discovered during the optimization to disk. If you start another optimization run this information will automatically be loaded into the short term memory.

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_iris
from hyperactive import Hyperactive

iris_data = load_iris()
X, y = iris_data.data, iris_data.target

def model(para, X, y):
    gbc = GradientBoostingClassifier(n_estimators=para["n_estimators"])
    return cross_val_score(gbc, X, y, cv=3).mean()

search_config = {model: {"n_estimators": range(5, 100)}}

'''
Saves the memory data after the optimization.
'''
opt = Hyperactive(X, y)
opt.search(search_config, n_iter=500, memory="long")

'''
The memory data is automatically loaded into short term memory.
'''
opt = Hyperactive(X, y, memory="long")
opt.search(search_config, n_iter=500)
```


## Memory helper functions
