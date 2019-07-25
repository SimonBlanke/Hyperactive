<h1 align="center">
Hyperactive
</h1>

<h3 align="center">A hyperparameter optimization toolbox for convenient and fast prototyping</h3>



<p align="center">
  
  <a href="https://travis-ci.com/SimonBlanke/Hyperactive">
    <img src="https://img.shields.io/travis/SimonBlanke/Hyperactive.svg?style=for-the-badge&logo=appveyor">
  </a>
 
  <a href="https://coveralls.io/github/SimonBlanke/Hyperactive">
    <img src="https://img.shields.io/coveralls/github/SimonBlanke/Hyperactive?style=for-the-badge&logo=appveyor">
  </a>
  
</p>




<br>

---

<p align="center">
  <a href="https://github.com/SimonBlanke/Hyperactive#overview">Overview</a> |
  <a href="https://github.com/SimonBlanke/Hyperactive#performance">Performance</a> |
  <a href="https://github.com/SimonBlanke/Hyperactive#installation">Installation</a> |
  <a href="https://github.com/SimonBlanke/Hyperactive#examples">Examples</a> |
  <a href="https://github.com/SimonBlanke/Hyperactive#advanced-features">Advanced Features</a> |
  <a href="https://github.com/SimonBlanke/Hyperactive#hyperactive-api">Hyperactive API</a> |
  <a href="https://github.com/SimonBlanke/Hyperactive#license-api">License</a>
</p>

---


<br>

## Overview

- Optimize hyperparameters of machine- or deep-learning models, using a simple API.
- Choose from a variety of different optimization techniques to improve your model.
- Never lose progress of previous optimizations: Just pass one or more models as start points and continue optimizing.
- Use transfer learning during the optimization process to build a more accurate model, while saving training and optimization time.
- Utilize multiprocessing for machine learning or your gpu for deep learning models.



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
        <a> <a href="https://github.com/SimonBlanke/Hyperactive/tree/master/hyperactive/optimizers#local-search">Local Search:</a> </a>
          <ul>
            <li>Hill Climbing</li>
            <li>Stochastic Hill Climbing</li>
            <li>Tabu Search</li>
         </ul>
        <a> <a href="https://github.com/SimonBlanke/Hyperactive/tree/master/hyperactive/optimizers#random-methods">Random Methods:</a> </a>
          <ul>
            <li>Random Search</li>
            <li>Random Restart Hill Climbing</li>
            <li>Random Annealing</li>
         </ul>
        <a> <a href="https://github.com/SimonBlanke/Hyperactive/tree/master/hyperactive/optimizers#markov-chain-monte-carlo">Markov Chain Monte Carlo:</a> </a>
          <ul>
            <li>Simulated Annealing</li>
            <li>Stochastic Tunneling</li>
            <li>Parallel Tempering</li>
          </ul>
        <a> <a href="https://github.com/SimonBlanke/Hyperactive/tree/master/hyperactive/optimizers#population-methods">Population Methods:</a> </a>
          <ul>
            <li>Particle Swarm Optimizer</li>
            <li>Evolution Strategy</li>
          </ul>
        <a> <a href="https://github.com/SimonBlanke/Hyperactive/tree/master/hyperactive/optimizers#sequential-methods">Sequential Methods:</a> </a>
          <ul>
            <li>Bayesian Optimization</li>
          </ul>
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


<br>

## Performance

The bar chart below shows, that the optimization process itself represents only a small fraction (<0.6%) of the computation time.
The 'No Opt'-bar shows the training time of a default Gradient-Boosting-Classifier normalized to 1. The other bars show the computation time relative to 'No Opt'. Each optimizer did 30 runs of 300 iterations, to get a good statistic.

<p align="center">
<img src="plots/optimizer_time.png" width="900"/>
</p>


<br>


## Installation

Hyperactive is developed and tested in python 3:

![https://pypi.org/project/hyperactive](https://img.shields.io/pypi/pyversions/hyperactive.svg?style=for-the-badge&logo=python&logoColor=white)

<br>

Hyperactive is available on PyPi:

![https://pypi.python.org/pypi/hyperactive](https://img.shields.io/pypi/v/hyperactive?style=for-the-badge&colorB=4cc61e) 
![https://pypi.python.org/pypi/hyperactive](https://img.shields.io/pypi/dm/hyperactive?style=for-the-badge)

```console
pip install hyperactive
```

<br>


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




<details><summary>Example with a multi-layer-perceptron in keras:</summary>
<p>

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from hyperactive import ParticleSwarmOptimizer

breast_cancer_data = load_breast_cancer()

X = breast_cancer_data.data
y = breast_cancer_data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# this defines the structure of the model and the search space in each layer
search_config = {
    "keras.compile.0": {"loss": ["binary_crossentropy"], "optimizer": ["adam"]},
    "keras.fit.0": {"epochs": [3], "batch_size": [100], "verbose": [0]},
    "keras.layers.Dense.1": {
        "units": range(5, 15),
        "activation": ["relu"],
        "kernel_initializer": ["uniform"],
    },
    "keras.layers.Dense.2": {
        "units": range(5, 15),
        "activation": ["relu"],
        "kernel_initializer": ["uniform"],
    },
    "keras.layers.Dense.3": {"units": [1], "activation": ["sigmoid"]},
}

Optimizer = ParticleSwarmOptimizer(
    search_config, n_iter=3, metric=["mean_absolute_error"], verbosity=0
)
# search best hyperparameter for given data
Optimizer.fit(X_train, y_train)
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




<br>


## Advanced Features

The features listed below can be activated during the instantiation of the optimizer ([see API](https://github.com/SimonBlanke/hyperactive#hyperactive-api)) and works with every optimizer in the hyperactive package.


#### [Memory](https://github.com/SimonBlanke/Hyperactive/blob/master/examples/example_memory.py)
After the evaluation of a model the position (in the hyperparameter search dictionary) and the cross-validation score are written to a dictionary. If the optimizer tries to evaluate this position again it can quickly lookup if a score for this position is present and use it instead of going through the extensive training and prediction process.


#### [Scatter-Initialization](https://github.com/SimonBlanke/Hyperactive/blob/master/examples/example_scatter_init.py)
This technique was inspired by the 'Hyperband Optimization' and aims to find a good initial position for the optimization. It does so by evaluating n random positions with a training subset of 1/n the size of the original dataset. The position that achieves the best score is used as the starting position for the optimization.


#### [Multiprocessing](https://github.com/SimonBlanke/Hyperactive/blob/master/examples/example_multiprocessing.py)
The multiprocessing in hyperactive works by creating additional searches, that run in parallel without any shared memory. This provides the possibility of hyperparameter-tuning of different models at the same time. If one single model should be tuned as fast as possible n_jobs in the optimizer should be set to '1', while n_jobs (of the model) in the search_config should be set to '-1'.

<details><summary>Two searches with eight cpu-cores:</summary>
<p>

```python
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from hyperactive import RandomSearchOptimizer

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
        "n_jobs": [4],
    },
    "sklearn.ensemble.GradientBoostingClassifier": {
        "n_estimators": range(10, 100, 10),
        "max_depth": range(1, 11),
        "min_samples_split": range(2, 21),
        "n_jobs": [4],
    },
}

Optimizer = RandomSearchOptimizer(search_config, n_iter=300, n_jobs=2, verbosity=0)

# search best hyperparameter for given data
Optimizer.fit(X_train, y_train)
```

</p>
</details>


<details><summary>One search with all cpu-cores:</summary>
<p>

```python
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from hyperactive import RandomSearchOptimizer

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
        "n_jobs": [-1],
    },
}

Optimizer = RandomSearchOptimizer(search_config, n_iter=300, n_jobs=1, verbosity=0)

# search best hyperparameter for given data
Optimizer.fit(X_train, y_train)
```

</p>
</details>




<details><summary>Multiple searches with all cpu-cores:</summary>
<p>

```python
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from hyperactive import RandomSearchOptimizer

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
    },
    "sklearn.neighbors.KNeighborsClassifier": {
        "n_neighbors": range(1, 10),
        "weights": ["uniform", "distance"],
        "p": [1, 2],
    },
    "sklearn.ensemble.GradientBoostingClassifier": {
        "n_estimators": range(10, 100, 10),
        "learning_rate": [1e-3, 1e-2, 1e-1, 0.5, 1.0],
        "max_depth": range(1, 11),
        "min_samples_split": range(2, 21),
        "min_samples_leaf": range(1, 21),
        "subsample": np.arange(0.05, 1.01, 0.05),
        "max_features": np.arange(0.05, 1.01, 0.05),
    },
    "sklearn.tree.DecisionTreeClassifier": {
        "criterion": ["gini", "entropy"],
        "max_depth": range(1, 11),
        "min_samples_split": range(2, 21),
        "min_samples_leaf": range(1, 21),
    },
}

Optimizer = RandomSearchOptimizer(search_config, n_iter=300, n_jobs=-1, verbosity=0)
```

</p>
</details>


#### [Transfer-Learning](https://github.com/SimonBlanke/Hyperactive/blob/master/examples/example_transfer_learning.py)
In the current implementation transfer-learning works by using a predefined model (with optional pretrained weights) provided by the keras package. The import path can be inserted as a layer (with its parameters in an sub-dictionary), like in a regular search dictionary. The following snippet provides an example:

<details><summary>Transfer-learning example:</summary>
<p>

```python
from keras.datasets import cifar10
from keras.utils import to_categorical

from hyperactive import SimulatedAnnealingOptimizer

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# this defines the structure of the model and the search space in each layer
search_config = {
    "keras.compile.0": {"loss": ["binary_crossentropy"], "optimizer": ["adam"]},
    "keras.fit.0": {"epochs": [1], "batch_size": [300], "verbose": [0]},
    # just add the pretrained model as a layer like this:
    "keras.applications.MobileNet.1": {
        "weights": ["imagenet"],
        "input_shape": [(32, 32, 3)],
        "include_top": [False],
    },
    "keras.layers.Flatten.2": {},
    "keras.layers.Dense.3": {
        "units": range(5, 15),
        "activation": ["relu"],
        "kernel_initializer": ["uniform"],
    },
    "keras.layers.Dense.4": {"units": [10], "activation": ["sigmoid"]},
}


Optimizer = SimulatedAnnealingOptimizer(
    search_config, n_iter=3, warm_start=False, verbosity=0
)
```

</p>
</details>


#### [Warm-Start](https://github.com/SimonBlanke/Hyperactive/blob/master/examples/example_warm_start_sklearn.py)

When a search is finished the warm-start-dictionary for the best position in the hyperparameter search space (and its metric) is printed in the command line (at verbosity=1). If multiple searches ran in parallel the warm-start-dictionaries are sorted by the best metric in decreasing order. If the start position in the warm-start-dictionary is not within the search space defined in the search_config an error will occure.

<details><summary>Warm-start example for sklearn model:</summary>
<p>

```python
start_point = {
    "sklearn.ensemble.RandomForestClassifier.0": {
        "n_estimators": [30],
        "max_depth": [6],
        "criterion": ["entropy"],
        "min_samples_split": [12],
        "min_samples_leaf": [16],
    },
    "sklearn.ensemble.RandomForestClassifier.1": {
        "n_estimators": [50],
        "max_depth": [3],
        "criterion": ["entropy"],
    },
}
```

</p>
</details>


<details><summary>Warm-start example for keras model (cnn):</summary>
<p>

```python
start_point = {
    "keras.compile.0": {"loss": ["categorical_crossentropy"], "optimizer": ["adam"]},
    "keras.fit.0": {"epochs": [3], "batch_size": [500], "verbose": [0]},
    "keras.layers.Conv2D.1": {
        "filters": [64],
        "kernel_size": [3],
        "activation": ["relu"],
        "input_shape": [(28, 28, 1)],
    },
    "keras.layers.MaxPooling2D.2": {"pool_size": [(2, 2)]},
    "keras.layers.Conv2D.3": {
        "filters": [32],
        "kernel_size": [3],
        "activation": ["relu"],
        "input_shape": [(28, 28, 1)],
    },
    "keras.layers.MaxPooling2D.4": {"pool_size": [(2, 2)]},
    "keras.layers.Conv2D.5": {
        "filters": [32],
        "kernel_size": [3],
        "activation": ["relu"],
        "input_shape": [(28, 28, 1)],
    },
    "keras.layers.MaxPooling2D.6": {"pool_size": [(2, 2)]},
    "keras.layers.Flatten.7": {},
    "keras.layers.Dense.8": {"units": [50], "activation": ["softmax"]},
    "keras.layers.Dropout.9": {"rate": [0.4]},
"keras.layers.Dense.10": {"units": [10], "activation": ["softmax"]},
}
```

</p>
</details>


<br>


## Hyperactive API

### Classes:
```python

HillClimbingOptimizer(search_config, n_iter, metric="accuracy", n_jobs=1, cv=5, verbosity=1, random_state=None, warm_start=False, memory=True, scatter_init=False, eps=1, r=1e-6)
StochasticHillClimbingOptimizer(search_config, n_iter, metric="accuracy", n_jobs=1, cv=5, verbosity=1, random_state=None, warm_start=False, memory=True, scatter_init=False)
TabuOptimizer(search_config, n_iter, metric="accuracy", n_jobs=1, cv=5, verbosity=1, random_state=None, warm_start=False, memory=True, scatter_init=False, eps=1, tabu_memory=[3, 6, 9])

RandomSearchOptimizer(search_config, n_iter, metric="accuracy", n_jobs=1, cv=5, verbosity=1, random_state=None, warm_start=False, memory=True, scatter_init=False)
RandomRestartHillClimbingOptimizer(search_config, n_iter, metric="accuracy", n_jobs=1, cv=5, verbosity=1, random_state=None, warm_start=False, memory=True, scatter_init=False, n_restarts=10)
RandomAnnealingOptimizer(search_config, n_iter, metric="accuracy", n_jobs=1, cv=5, verbosity=1, random_state=None, warm_start=False, memory=True, scatter_init=False, eps=100, t_rate=0.98)

SimulatedAnnealingOptimizer(search_config, n_iter, metric="accuracy", n_jobs=1, cv=5, verbosity=1, random_state=None, warm_start=False, memory=True, scatter_init=False, eps=1, t_rate=0.98)
StochasticTunnelingOptimizer(search_config, n_iter, metric="accuracy", n_jobs=1, cv=5, verbosity=1, random_state=None, warm_start=False, memory=True, scatter_init=False, eps=1, t_rate=0.98, n_neighbours=1, gamma=1)
ParallelTemperingOptimizer(search_config, n_iter, metric="accuracy", n_jobs=1, cv=5, verbosity=1, random_state=None, warm_start=False, memory=True, scatter_init=False, eps=1, t_rate=0.98, n_neighbours=1, system_temps=[0.1, 0.2, 0.01], n_swaps=10)

ParticleSwarmOptimizer(search_config, n_iter, metric="accuracy", n_jobs=1, cv=5, verbosity=1, random_state=None, warm_start=False, memory=True, scatter_init=False, n_part=4, w=0.5, c_k=0.5, c_s=0.9)
EvolutionStrategyOptimizer(search_config, n_iter, metric="accuracy", n_jobs=1, cv=5, verbosity=1, random_state=None, warm_start=False, memory=True, scatter_init=False, individuals=10, mutation_rate=0.7, crossover_rate=0.3)

BayesianOptimizer(search_config, n_iter, metric="accuracy", n_jobs=1, cv=5, verbosity=1, random_state=None, warm_start=False, memory=True, scatter_init=False)

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
| scatter_init  |  int | False  |  Chooses better initial position by training on multiple random positions with smaller training dataset (split into int subsets)  |


### Specific keyword arguments (hill climbing):

| Argument | Type | Default | Description |
| ------ | ------ | ------ | ------ |
| eps  | int | 1 | epsilon |


### Specific keyword arguments (stochastic hill climbing):

| Argument | Type | Default | Description |
| ------ | ------ | ------ | ------ |
| eps  | int | 1 | epsilon |
|  r | float  |  1e-6 | acceptance factor  |


### Specific keyword arguments (tabu search):

| Argument | Type | Default | Description |
| ------ | ------ | ------ | ------ |
| eps  | int | 1 | epsilon |
|  tabu_memory | list  |  [3, 6, 9] | length of short/mid/long-term memory  |

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


### Specific keyword arguments (parallel tempering):

| Argument | Type | Default | Description |
| ------ | ------ | ------ | ------ |
| eps  | int | 1 | epsilon |
| t_rate | float | 0.98 | cooling rate  |
| system_temps  | list  |  [0.1, 0.2, 0.01] | initial temperatures (number of elements defines number of systems)  |
|  n_swaps | int  | 10  | number of swaps  |


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

<br>


## License

![https://github.com/SimonBlanke/Hyperactive/blob/master/LICENSE](https://img.shields.io/github/license/SimonBlanke/Hyperactive?style=for-the-badge)

