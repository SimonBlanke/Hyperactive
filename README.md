# Hyperactive
A Python package for meta-heuristic hyperparameter optimization of scikit-learn models for supervised learning. Hyperactive automates the search for hyperparameters by utilizing metaheuristics to efficiently explore the search space and provide a sufficiently good solution. Its API is similar to scikit-learn and allows for parallel computation. Hyperactive offers a small collection of the following meta-heuristic optimization techniques:
  - Random search
  - Simulated annealing
  - Particle swarm optimization

### Installation
```python
pip install hyperactive
```

### Example
```python
from sklearn.datasets import load_iris
from hyperactive import SimulatedAnnealing_Optimizer

iris_data = load_iris()
iris_X_train = iris_data.data
iris_y_train = iris_data.target

search_dict = {
    'sklearn.ensemble.RandomForestClassifier': {
        'n_estimators': [200],
        'criterion': ["gini", "entropy"],
        'min_samples_split': range(2, 21),
        'min_samples_leaf':  range(2, 21),
      }
}

Optimizer = SimulatedAnnealing_Optimizer(search_dict, n_searches=1000, scoring='accuracy', n_jobs=1)
Optimizer.fit(iris_X_train, iris_y_train)
```

### Implementation

### Performance comparison
