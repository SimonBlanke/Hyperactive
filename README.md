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

```python
Usable classes:
    - RandomSearch_Optimizer(search_dict, n_iter, scoring='accuracy', n_jobs=1, cv=5)
    - SimulatedAnnealing_Optimizer(search_dict, n_iter, scoring='accuracy', eps=1, t_rate=0.9, n_jobs=1, cv=5)
    - ParticleSwarm_Optimizer(search_dict, n_iter, scoring='accuracy', n_part=1, w=0.5, c_k=0.8, c_s=0.9, n_jobs=1, cv=5)

Methods:
    - fit(X_train, y_train)
    - predict(X_test)
    - score(X_test, y_test)
    - export(filename, path="")
```
