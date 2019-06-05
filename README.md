[![PyPI version](https://img.shields.io/pypi/v/hyperactive.svg)](https://pypi.python.org/pypi/hyperactive)
[![PyPI license](https://img.shields.io/pypi/l/hyperactive.svg)](https://github.com/SimonBlanke/hyperactive/blob/master/LICENSE)


# Hyperactive
A Python package for meta-heuristic hyperparameter optimization of scikit-learn models for supervised learning. Hyperactive automates the search for hyperparameters by utilizing metaheuristics to efficiently explore the search space and provide a sufficiently good solution. Its API is similar to scikit-learn and allows for parallel computation. Hyperactive offers a small collection of the following meta-heuristic optimization techniques:
  - Random search
  - Simulated annealing
  - Particle swarm optimization

The multiprocessing will start n_jobs separate searches. These can operate independent of one another, which makes the workload perfectly parallel. In the current implementation the actual number of searches in each process is n_iter divided by n_jobs and rounded down to the next integer.


## Installation
```console
pip install hyperactive
```


## Example
```python
from sklearn.datasets import load_iris
from hyperactive import SimulatedAnnealing_Optimizer

iris_data = load_iris()
X_train = iris_data.data
y_train = iris_data.target

search_dict = {
    'sklearn.ensemble.RandomForestClassifier': {
        'n_estimators': [100],
        'criterion': ["gini", "entropy"],
        'min_samples_split': range(2, 21),
        'min_samples_leaf':  range(2, 21),
      }
}

Optimizer = SimulatedAnnealing_Optimizer(search_dict, n_iter=1000, scoring='accuracy', n_jobs=2)
Optimizer.fit(X_train, y_train)
```


## Hyperactive API

### Classes:
```python
RandomSearch_Optimizer(search_space, n_iter, scoring="accuracy", n_jobs=1, cv=5, verbosity=1, random_state=None, start_points=None)
SimulatedAnnealing_Optimizer(search_space, n_iter, scoring="accuracy", n_jobs=1, cv=5, verbosity=1, random_state=None, start_points=None, eps=1, t_rate=0.95)
ParticleSwarm_Optimizer(search_space, n_iter, scoring="accuracy", n_jobs=1, cv=5, verbosity=1, random_state=None, start_points=None, n_part=2, w=0.5, c_k=0.5, c_s=0.9)
```

### General positional argument:

| Argument | Type | Description |
| ------ | ------ | ------ |
| search_space  | dict | bla |
| n_iter | int | bla |

### General keyword arguments:

| Argument | Type | Default | Description |
| ------ | ------ | ------ | ------ |
| scoring  | str | "accuracy" | bla |
| n_jobs | int | bla | bla |
| cv | int | bla | bla |
| verbosity | int | bla | bla |
| random_state | int | bla | bla |
| start_points | dict | bla | bla |

### Specific keyword arguments (simulated annealing):

| Argument | Type | Default | Description |
| ------ | ------ | ------ | ------ |
| eps  | int | 1 | bla |
| t_rate | float | 0.95 | bla |

### Specific keyword arguments (particle swarm optimization):

| Argument | Type | Default | Description |
| ------ | ------ | ------ | ------ |
| n_part  | int | 1 | bla |
| w | float | 0.5 | bla |
| c_k | float | 0.8 | bla |
| c_s | float | 0.9 | bla |

### General methods:
```
fit(self, X_train, y_train)
```
| Argument | Type | Description |
| ------ | ------ | ------ |
| X_train  | array-like | training input features |
| y_train | array-like | training target |

```
predict(self, X_test)
```
| Argument | Type | Description |
| ------ | ------ | ------ |
| X_test  | array-like | testing input features |

```
score(self, X_test, y_test)
```
| Argument | Type | Description |
| ------ | ------ | ------ |
| X_test  | array-like | testing input features |
| y_test | array-like | true values |

```
export(self, filename)
```
| Argument | Type | Description |
| ------ | ------ | ------ |
| filename  | str | file name and path for model export |
