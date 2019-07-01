<h1 align="center">
Hyperactive
</h1>

<h3 align="center">A hyperparameter optimization toolbox for convenient and fast prototyping.</h3>

---

<p align="center">

  <a href="https://pypi.python.org/pypi/hyperactive">
    <img src="https://img.shields.io/pypi/v/hyperactive.svg">
  </a>

  <a href="https://github.com/SimonBlanke/hyperactive/blob/master/LICENSE">
    <img src="https://img.shields.io/pypi/l/hyperactive.svg">
  </a>

  <a href="https://pepy.tech/project/hyperactive">
    <img src="https://pepy.tech/badge/hyperactive">
  </a>
  <a href="https://github.com/python/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg">
  </a>

</p>



## Overview:
- Optimize hyperparameters of machine- or deep-learning models from: 
    - scikit-learn
    - xgboost
    - keras
- Choose from a variety of different optimization techniques to improve your model, including:
    - Random search
    - Simulated annealing
    - Particle swarm optimization
    - Evolution strategy
- Never lose progress of previous optimizations: Just pass one or more models as start points and continue optimizing
- Use transfer learning during the optimization process to build a more accurate model, while saving training and optimization time
- Utilize multiprocessing for machine learning or your gpu for deep learning models


---

<p align="center">
  <a href="https://github.com/SimonBlanke/hyperactive#installation">Installation</a> |
  <a href="https://github.com/SimonBlanke/hyperactive#examples">Examples</a> |
  <a href="https://github.com/SimonBlanke/hyperactive#hyperactive-api">Hyperactive API</a>
</p>

---

## Installation
```console
pip install hyperactive
```


## Examples

Basic sklearn example:
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from hyperactive import SimulatedAnnealing_Optimizer

iris_data = load_iris()
X = iris_data.data
y = iris_data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# this defines the model and hyperparameter search space
search_config = {
    "sklearn.ensemble.RandomForestClassifier": {
        "n_estimators": range(10, 100, 10),
        "max_depth": [3, 4, 5, 6],
        "criterion": ["gini", "entropy"],
        "min_samples_split": range(2, 21),
        "min_samples_leaf": range(2, 21),
    }
}

Optimizer = SimulatedAnnealing_Optimizer(search_config, n_iter=100, n_jobs=4)

# search best hyperparameter for given data
Optimizer.fit(X_train, y_train)

# predict from test data
prediction = Optimizer.predict(X_test)

# calculate accuracy score
score = Optimizer.score(X_test, y_test)
```

Example with a convolutional neural network in keras:
```python
import numpy as np

from keras.datasets import mnist
from keras.utils import to_categorical

from hyperactive import RandomSearch_Optimizer

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# this defines the structure of the model and the search space in each layer
search_config = {
    "keras.compile.0": {"loss": ["categorical_crossentropy"], "optimizer": ["adam"]},
    "keras.fit.0": {"epochs": [20], "batch_size": [500], "verbose": [2]},
    "keras.layers.Conv2D.1": {
        "filters": [32, 64, 128],
        "kernel_size": range(3, 4),
        "activation": ["relu"],
        "input_shape": [(28, 28, 1)],
    },
    "keras.layers.MaxPooling2D.2": {"pool_size": [(2, 2)]},
    "keras.layers.Conv2D.3": {
        "filters": [16, 32, 64],
        "kernel_size": [3],
        "activation": ["relu"],
    },
    "keras.layers.MaxPooling2D.4": {"pool_size": [(2, 2)]},
    "keras.layers.Flatten.5": {},
    "keras.layers.Dense.6": {"units": range(30, 200, 10), "activation": ["softmax"]},
    "keras.layers.Dropout.7": {"rate": list(np.arange(0.4, 0.8, 0.1))},
    "keras.layers.Dense.8": {"units": [10], "activation": ["softmax"]},
}

Optimizer = RandomSearch_Optimizer(search_config, n_iter=20)

# search best hyperparameter for given data
Optimizer.fit(X_train, y_train)

# predict from test data
prediction = Optimizer.predict(X_test)

# calculate accuracy score
score = Optimizer.score(X_test, y_test)
```




## Hyperactive API

### Classes:
```python
RandomSearch_Optimizer(search_config, n_iter, metric="accuracy", n_jobs=1, cv=5, verbosity=1, random_state=None, warm_start=False)
SimulatedAnnealing_Optimizer(search_config, n_iter, metric="accuracy", n_jobs=1, cv=5, verbosity=1, random_state=None, warm_start=False, eps=1, t_rate=0.99)
ParticleSwarm_Optimizer(search_config, n_iter, metric="accuracy", n_jobs=1, cv=5, verbosity=1, random_state=None, warm_start=False, n_part=1, w=0.5, c_k=0.5, c_s=0.9)
EvolutionStrategy_Optimizer(search_config, n_iter, metric="accuracy", memory=None, n_jobs=1, cv=5, verbosity=1, random_state=None, warm_start=False, individuals=10, mutation_rate=0.7, crossover_rate=0.3)

```

### General positional argument:

| Argument | Type | Description |
| ------ | ------ | ------ |
| search_config  | dict | hyperparameter search space to explore by the optimizer |
| n_iter | int | number of iterations to perform |

### General keyword arguments:

| Argument | Type | Default | Description |
| ------ | ------ | ------ | ------ |
| metric  | str | "accuracy" | metric for model evaluation |
| n_jobs | int | 1 | number of jobs to run in parallel (-1 for maximum) |
| cv | int | 5 | cross-validation |
| verbosity | int | 1 | Shows model and metric information |
| random_state | int | None | The seed for random number generator |
| warm_start | dict | None | Hyperparameter configuration to start from |

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

### Specific keyword arguments (evolution strategy optimization):

| Argument | Type | Default | Description |
| ------ | ------ | ------ | ------ |
| individuals  | int | 10 | number of individuals |
| mutation_rate | float | 0.7 | mutation rate |
| crossover_rate | float | 0.3 | crossover rate |

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
