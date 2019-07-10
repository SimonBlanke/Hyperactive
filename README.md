<h1 align="center">
Hyperactive
</h1>

<h3 align="center">A hyperparameter optimization toolbox for convenient and fast prototyping.</h3>


<p align="center">

  <a href="https://pypi.python.org/pypi/hyperactive">
    <img src="https://img.shields.io/pypi/v/hyperactive.svg?colorB=4cc61e">
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


---

<p align="center">
  <a href="https://github.com/SimonBlanke/hyperactive#overview">Overview</a> |
  <a href="https://github.com/SimonBlanke/hyperactive#performance">Performance</a> |
  <a href="https://github.com/SimonBlanke/hyperactive#installation">Installation</a> |
  <a href="https://github.com/SimonBlanke/hyperactive#examples">Examples</a> |
  <a href="https://github.com/SimonBlanke/hyperactive#hyperactive-api">Hyperactive API</a>
</p>

---



## Overview:
- Optimize hyperparameters of machine- or deep-learning models
- Choose from a variety of different optimization techniques to improve your model
- Never lose progress of previous optimizations: Just pass one or more models as start points and continue optimizing
- Use transfer learning during the optimization process to build a more accurate model, while saving training and optimization time
- Utilize multiprocessing for machine learning or your gpu for deep learning models



<table>
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Optimization Techniques</b>
        <img src="images/blue.jpg"/>
      </td>
      <td>
        <b>Supported Packages</b>
        <img src="images/blue.jpg"/>
      </td>
    </tr>
    <tr/>
    <tr valign="top">
      <td>
        <a>Local Search:</a>
          <ul>
            <li>Hill Climbing</li>
            <li>Stochastic Hill Climbing</li>
         </ul>
        <a>Random Methods:</a>
          <ul>
            <li>Random Search</li>
            <li>Random Restart Hill Climbing</li>
            <li>Random Annealing</li>
         </ul>
        <a>Markov Chain Monte Carlo:</a>
          <ul>
            <li>Simulated Annealing</li>
            <li>Stochastic Tunneling</li>
          </ul>
        <a>Population Methods:</a>
          <ul>
            <li>Particle Swarm Optimizer</li>
            <li>Evolution Strategy</li>
      </td>
      <td>
        <a>Machine Learning:</a>
          <ul>
              <li>Scikit-learn</li>
              <li>XGBoost</li>
          </ul>
        <a>Deep Learning:</a>
          <ul>
              <li>Keras</li>
          </ul>
        <a>Distribution:</a>
          <ul>
              <li>Multiprocessing</li>
          </ul>
      </td>
    </tr>
  </tbody>
</table>


## Performance

The bar chart below shows, that the optimization process itself represents only a small fraction (<0.6%) of the computation time. 
The 'No Opt'-bar shows the training time of a default Gradient-Boosting-Classifier normalized to 1. The other bars show the computation time relative to 'No Opt'. Each optimizer did 30 runs of 300 iterations, to get a good statistic. 

<p align="center">
<img src="plots/optimizer_time.png" width="900"/>
</p>


## Installation

```console
pip install hyperactive
```


## Examples

<details><summary>Basic sklearn example:</summary>
<p>

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from hyperactive import SimulatedAnnealingOptimizer

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

Optimizer = SimulatedAnnealingOptimizer(search_config, n_iter=100, n_jobs=4)

# search best hyperparameter for given data
Optimizer.fit(X_train, y_train)

# predict from test data
prediction = Optimizer.predict(X_test)

# calculate accuracy score
score = Optimizer.score(X_test, y_test)
```

</p>
</details>







<details><summary>Example with a convolutional neural network in keras:</summary>
<p>

```python
import numpy as np

from keras.datasets import mnist
from keras.utils import to_categorical

from hyperactive import RandomSearchOptimizer

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

Optimizer = RandomSearchOptimizer(search_config, n_iter=20)

# search best hyperparameter for given data
Optimizer.fit(X_train, y_train)

# predict from test data
prediction = Optimizer.predict(X_test)

# calculate accuracy score
score = Optimizer.score(X_test, y_test)
```

</p>
</details>




## Hyperactive API

### Classes:
```python

HillClimbingOptimizer(search_config, n_iter, metric="accuracy", n_jobs=1, cv=5, verbosity=1, random_state=None, warm_start=False, memory=True, hyperband_init=False, eps=1, r=1e-6)
StochasticHillClimbingOptimizer(search_config, n_iter, metric="accuracy", n_jobs=1, cv=5, verbosity=1, random_state=None, warm_start=False, memory=True, hyperband_init=False,)
RandomSearchOptimizer(search_config, n_iter, metric="accuracy", n_jobs=1, cv=5, verbosity=1, random_state=None, warm_start=False, memory=True, hyperband_init=False)
RandomRestartHillClimbingOptimizer(search_config, n_iter, metric="accuracy", n_jobs=1, cv=5, verbosity=1, random_state=None, warm_start=False, memory=True, hyperband_init=False, n_restarts=10)
RandomAnnealingOptimizer(search_config, n_iter, metric="accuracy", n_jobs=1, cv=5, verbosity=1, random_state=None, warm_start=False, memory=True, hyperband_init=False, eps=100, t_rate=0.98)
SimulatedAnnealingOptimizer(search_config, n_iter, metric="accuracy", n_jobs=1, cv=5, verbosity=1, random_state=None, warm_start=False, memory=True, hyperband_init=False, eps=1, t_rate=0.98)
StochasticTunnelingOptimizer(search_config, n_iter, metric="accuracy", n_jobs=1, cv=5, verbosity=1, random_state=None, warm_start=False, memory=True, hyperband_init=False, eps=1, t_rate=0.98, n_neighbours=1, gamma=1)
ParticleSwarmOptimizer(search_config, n_iter, metric="accuracy", n_jobs=1, cv=5, verbosity=1, random_state=None, warm_start=False, memory=True, hyperband_init=False, n_part=4, w=0.5, c_k=0.5, c_s=0.9)
EvolutionStrategyOptimizer(search_config, n_iter, metric="accuracy", n_jobs=1, cv=5, verbosity=1, random_state=None, warm_start=False, memory=True, hyperband_init=False, individuals=10, mutation_rate=0.7, crossover_rate=0.3)

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
| memory  |  bool | True  |  Stores explored evaluations in a dictionary to save computing time |
| hyperband_init  |  int | False  |  Chooses better initial position by training on multiple random positions with smaller training dataset (split into int subsets)  |


### Specific keyword arguments (hill climbing):

| Argument | Type | Default | Description |
| ------ | ------ | ------ | ------ |
| eps  | int | 1 | epsilon |


### Specific keyword arguments (stochastic hill climbing):

| Argument | Type | Default | Description |
| ------ | ------ | ------ | ------ |
| eps  | int | 1 | epsilon |
|  r | float  |  1e-6 | acceptance factor  |

### Specific keyword arguments (random restart hill climbing):

| Argument | Type | Default | Description |
| ------ | ------ | ------ | ------ |
| eps  | int | 1 | epsilon |
| n_restarts  | int  | 10  | number of restarts  |


### Specific keyword arguments (random annealing):

| Argument | Type | Default | Description |
| ------ | ------ | ------ | ------ |
| eps  | int | 100 | epsilon |
| t_rate | float | 0.98 | cooling rate  |

### Specific keyword arguments (simulated annealing):

| Argument | Type | Default | Description |
| ------ | ------ | ------ | ------ |
| eps  | int | 1 | epsilon |
| t_rate | float | 0.98 | cooling rate  |

### Specific keyword arguments (stochastic tunneling):

| Argument | Type | Default | Description |
| ------ | ------ | ------ | ------ |
| eps  | int | 1 | epsilon |
| t_rate | float | 0.98 | cooling rate  |
| gamma  | float  |  1 | tunneling factor  |


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
