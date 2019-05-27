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
