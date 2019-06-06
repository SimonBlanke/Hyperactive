[![PyPI version](https://img.shields.io/pypi/v/hyperactive.svg)](https://pypi.python.org/pypi/hyperactive)
[![PyPI license](https://img.shields.io/pypi/l/hyperactive.svg)](https://github.com/SimonBlanke/hyperactive/blob/master/LICENSE)
[![PyPI downloads](https://img.shields.io/pypi/dm/hyperactive.svg)](https://pypi.python.org/pypi/hyperactive)



# Hyperactive
A Python package for meta-heuristic hyperparameter optimization of scikit-learn models for supervised learning. Hyperactive automates the search for hyperparameters by utilizing metaheuristics to efficiently explore the search space and provide a sufficiently good solution. Its API is similar to scikit-learn and allows for parallel computation. Hyperactive offers a small collection of the following meta-heuristic optimization techniques:
  - Random search
  - Simulated annealing
  - Particle swarm optimization

The multiprocessing will start n_jobs separate searches. These can operate independent of one another, which makes the workload perfectly parallel.


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
SimulatedAnnealing_Optimizer(search_space, n_iter, scoring="accuracy", n_jobs=1, cv=5, verbosity=1, random_state=None, start_points=None, eps=1, t_rate=0.99)
ParticleSwarm_Optimizer(search_space, n_iter, scoring="accuracy", n_jobs=1, cv=5, verbosity=1, random_state=None, start_points=None, n_part=1, w=0.5, c_k=0.5, c_s=0.9)
```

### General positional argument:

| Argument | Type | Description |
| ------ | ------ | ------ |
| search_space  | dict | hyperparameter search space to explore by the optimizer |
| n_iter | int | number of iterations to perform |

### General keyword arguments:

| Argument | Type | Default | Description |
| ------ | ------ | ------ | ------ |
| scoring  | str | "accuracy" | scoring for model evaluation |
| n_jobs | int | 1 | number of jobs to run in parallel (-1 for maximum) |
| cv | int | 5 | cross-validation |
| verbosity | int | 1 | Shows model and scoring information |
| random_state | int | None | The seed for random number generator |
| start_points | dict | None | Hyperparameter configuration to start from |

### Specific keyword arguments (simulated annealing):

| Argument | Type | Default | Description |
| ------ | ------ | ------ | ------ |
| eps  | int | 1 | epsilon |
| t_rate | float | 0.99 | cooling rate  |

### Specific keyword arguments (particle swarm optimization):

| Argument | Type | Default | Description |
| ------ | ------ | ------ | ------ |
| n_part  | int | 1 | number of particles |
| w | float | 0.5 | intertia factor |
| c_k | float | 0.8 | cognitive factor |
| c_s | float | 0.9 | social factor |

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
