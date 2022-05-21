<p align="center">
  <a href="https://github.com/SimonBlanke/Hyperactive"><img src="./docs/images/logo.png" height="250"></a>
</p>

<br>

---

<p align="center">
  <a href="https://github.com/SimonBlanke/Hyperactive/actions">
            <img src="https://github.com/SimonBlanke/Hyperactive/actions/workflows/tests.yml/badge.svg?branch=master" alt="img not loaded: try F5 :)">
          </a>
  <a href="https://app.codecov.io/gh/SimonBlanke/Hyperactive">
    <img src="https://img.shields.io/codecov/c/github/SimonBlanke/Hyperactive/master&logo=codecov" alt="img not loaded: try F5 :)">
  </a>
  <a href="https://codeclimate.com/github/SimonBlanke/Hyperactive">
  <img src="https://img.shields.io/codeclimate/maintainability/SimonBlanke/Hyperactive?style=flat-square&logo=code-climate" alt="img not loaded: try F5 :)">
  </a>
  <a href="https://scrutinizer-ci.com/g/SimonBlanke/Hyperactive/">
  <img src="https://img.shields.io/scrutinizer/quality/g/SimonBlanke/Hyperactive?style=flat-square&logo=scrutinizer-ci" alt="img not loaded: try F5 :)">
  </a>
  <a href="https://pypi.org/project/hyperactive/">
  <img src="https://img.shields.io/pypi/v/Hyperactive?style=flat-square&logo=PyPi&logoColor=white" alt="img not loaded: try F5 :)">
  </a>
  <a href="https://pypi.org/project/hyperactive/">
  <img src="https://img.shields.io/pypi/pyversions/hyperactive.svg?style=flat-square&logo=Python&logoColor=white" alt="img not loaded: try F5 :)">
  </a>
</p>

<h2 align="center">An optimization and data collection toolbox for convenient and fast prototyping of computationally expensive models.</h2>

<br>

<img src="./docs/images/bayes_convex.gif" align="right" width="500">

## Hyperactive:

- is [very easy](#hyperactive-is-very-easy-to-use) to learn but [extremly versatile](./examples/optimization_applications/search_space_example.py)

- provides intelligent [optimization algorithms](#overview), support for all mayor [machine-learning frameworks](#overview) and many interesting [applications](#overview)

- makes optimization [data collection](./examples/optimization_applications/meta_data_collection.py) simple

- [visualizes](https://github.com/SimonBlanke/ProgressBoard) your collected data

- saves your [computation time](./examples/optimization_applications/memory.py)

- supports [parallel computing](./examples/tested_and_supported_packages/multiprocessing_example.py)


<br>
<br>
<br>

---

<div align="center"><a name="menu"></a>
  <h3>
    <a href="https://github.com/SimonBlanke/Hyperactive#overview">Overview</a> •
    <a href="https://github.com/SimonBlanke/Hyperactive#installation">Installation</a> •
    <a href="https://github.com/SimonBlanke/Hyperactive#hyperactive-api-reference">API reference</a> •
    <a href="https://github.com/SimonBlanke/Hyperactive#roadmap">Roadmap</a> •
    <a href="https://github.com/SimonBlanke/Hyperactive#citing-hyperactive">Citation</a> •
    <a href="https://github.com/SimonBlanke/Hyperactive#license">License</a>
  </h3>
</div>

---

<br>

## What's new?
  - ### 04.05.2022 v4.2.0 with support of handling Exceptions and Callbacks
  - ### 28.01.2022 New implementation of the [Hyperactive ProgressBoard](https://github.com/SimonBlanke/ProgressBoard)


<br>

## Overview

<h3 align="center">
Hyperactive features a collection of optimization algorithms that can be used for a variety of optimization problems. The following table shows examples of its capabilities:
</h3>


<br>

<table>
  <tbody>
    <tr align="center" valign="center">
      <td>
        <strong>Optimization Techniques</strong>
        <img src="./docs/images/blue.jpg"/>
      </td>
      <td>
        <strong>Tested and Supported Packages</strong>
        <img src="./docs/images/blue.jpg"/>
      </td>
      <td>
        <strong>Optimization Applications</strong>
        <img src="./docs/images/blue.jpg"/>
      </td>
    </tr>
    <tr/>
    <tr valign="top">
      <td>
        <a><b>Local Search:</b></a>
          <ul>
            <li><a href="./examples/optimization_techniques/hill_climbing.py">Hill Climbing</a></li>
            <li><a href="./examples/optimization_techniques/repulsing_hill_climbing.py">Repulsing Hill Climbing</a></li>
            <li><a href="./examples/optimization_techniques/simulated_annealing.py">Simulated Annealing</a></li>
            <li>Downhill Simplex Optimizer</li>
         </ul><br>
        <a><b>Global Search:</b></a>
          <ul>
            <li><a href="./examples/optimization_techniques/random_search.py">Random Search</a></li>
            <li>Grid Search Optimizer</li>
            <li><a href="./examples/optimization_techniques/rand_rest_hill_climbing.py">Random Restart Hill Climbing</a></li>
            <li><a href="./examples/optimization_techniques/random_annealing.py">Random Annealing</a> [<a href="#/./overview#experimental-algorithms">*</a>] </li>
            <li>Powell's Method</li>
            <li>Pattern Search</li>
         </ul><br>
        <a><b>Population Methods:</b></a>
          <ul>
            <li><a href="./examples/optimization_techniques/parallel_tempering.py">Parallel Tempering</a></li>
            <li><a href="./examples/optimization_techniques/particle_swarm_optimization.py">Particle Swarm Optimizer</li>
            <li><a href="./examples/optimization_techniques/evolution_strategy.py">Evolution Strategy</a></li>
          </ul><br>
        <a><b>Sequential Methods:</b></a>
          <ul>
            <li><a href="./examples/optimization_techniques/bayesian_optimization.py">Bayesian Optimization</a></li>
            <li><a href="./examples/optimization_techniques/tpe.py">Tree of Parzen Estimators</a></li>
            <li><a href="./examples/optimization_techniques/forest_optimization.py">Forest Optimizer</a>
            [<a href="#/./overview#references">dto</a>] </li>
          </ul>
      </td>
      <td>
        <a><b>Machine Learning:</b></a>
          <ul>
              <li><a href="./examples/tested_and_supported_packages/sklearn_example.py">Scikit-learn</a></li>
              <li><a href="./examples/tested_and_supported_packages/xgboost_example.py">XGBoost</a></li>
              <li><a href="./examples/tested_and_supported_packages/lightgbm_example.py">LightGBM</a></li>
              <li><a href="./examples/tested_and_supported_packages/catboost_example.py">CatBoost</a></li>
              <li><a href="./examples/tested_and_supported_packages/rgf_example.py">RGF</a></li>
              <li><a href="./examples/tested_and_supported_packages/mlxtend_example.py">Mlxtend</a></li>
          </ul><br>
        <a><b>Deep Learning:</b></a>
          <ul>
              <li><a href="./examples/tested_and_supported_packages/tensorflow_example.py">Tensorflow</a></li>
              <li><a href="./examples/tested_and_supported_packages/keras_example.py">Keras</a></li>
              <li><a href="./examples/tested_and_supported_packages/pytorch_example.py">Pytorch</a></li>
          </ul><br>
        <a><b>Parallel Computing:</b></a>
          <ul>
              <li><a href="./examples/tested_and_supported_packages/multiprocessing_example.py">Multiprocessing</a></li>
              <li><a href="./examples/tested_and_supported_packages/joblib_example.py">Joblib</a></li>
              <li>Pathos</li>
          </ul>
      </td>
      <td>
        <a><b>Feature Engineering:</b></a>
          <ul>
            <li><a href="./examples/optimization_applications/feature_transformation.py">Feature Transformation</a></li>
            <li><a href="./examples/optimization_applications/feature_selection.py">Feature Selection</a></li>
            <li>Feature Construction</li>
          </ul>
        <a><b>Machine Learning:</b></a>
          <ul>
            <li><a href="./examples/optimization_applications/hyperpara_optimize.py">Hyperparameter Tuning</a></li>
            <li><a href="./examples/optimization_applications/model_selection.py">Model Selection</a></li>
            <li><a href="./examples/optimization_applications/sklearn_pipeline_example.py">Sklearn Pipelines</a></li>
            <li><a href="./examples/optimization_applications/ensemble_learning_example.py">Ensemble Learning</a></li>
          </ul>
        <a><b>Deep Learning:</b></a>
          <ul>
            <li><a href="./examples/optimization_applications/neural_architecture_search.py">Neural Architecture Search</a></li>
            <li><a href="./examples/optimization_applications/pretrained_nas.py">Pretrained Neural Architecture Search</a></li>
            <li><a href="./examples/optimization_applications/transfer_learning.py">Transfer Learning</a></li>
          </ul>
        <a><b>Data Collection:</b></a>
          <ul>
            <li><a href="./examples/optimization_applications/meta_data_collection.py">Search Data Collection</a></li>
            <li><a href="./examples/optimization_applications/meta_optimization.py">Meta Optimization</a></li>
            <li><a href="./examples/optimization_applications/meta_learning.py">Meta Learning</a></li>
          </ul>
        <a><b>Visualization:</b></a>
          <ul>
            <li><a href="./examples/optimization_applications/progress_visualization.py">Optimization progress visualization  </a></li>
          </ul>
        <a><b>Miscellaneous:</b></a>
          <ul>
            <li><a href="./examples/optimization_applications/test_function.py">Test Functions</a></li>
            <li>Fit Gaussian Curves</li>
            <li><a href="./examples/optimization_applications/multiple_scores.py">Managing multiple objectives</a></li>
            <li><a href="./examples/optimization_applications/search_space_example.py">Managing objects in search space</a></li>
            <li><a href="./examples/optimization_applications/memory.py">Memorize evaluations</a></li>
          </ul>
      </td>
    </tr>
  </tbody>
</table>

The examples above are not necessarily done with realistic datasets or training procedures. 
The purpose is fast execution of the solution proposal and giving the user ideas for interesting usecases.

<br>


### Hyperactive is very easy to use:

<table>
<tbody>
<tr>
<th> Regular training </th>
<th> Hyperactive </th>
</tr>
<tr>
<td>

```python
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import fetch_california_housing


data = fetch_california_housing()
X, y = data.data, data.target


gbr = DecisionTreeRegressor(max_depth=10)
score = cross_val_score(gbr, X, y, cv=3).mean()









```

</td>
<td>

```python
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import fetch_california_housing
from hyperactive import Hyperactive

data = fetch_california_housing()
X, y = data.data, data.target

def model(opt):
    gbr = DecisionTreeRegressor(max_depth=opt["max_depth"])
    return cross_val_score(gbr, X, y, cv=3).mean()


search_space = {"max_depth": list(range(3, 25))}

hyper = Hyperactive()
hyper.add_search(model, search_space, n_iter=50)
hyper.run()
```

</td>
</tr>
</tbody>
</table>



<br>

## Installation

The most recent version of Hyperactive is available on PyPi:

[![pyversions](https://img.shields.io/pypi/pyversions/hyperactive.svg?style=for-the-badge&logo=python&color=blue&logoColor=white)](https://pypi.org/project/hyperactive)
[![PyPI version](https://img.shields.io/pypi/v/hyperactive?style=for-the-badge&logo=pypi&color=green&logoColor=white)](https://pypi.org/project/hyperactive/)
[![PyPI version](https://img.shields.io/pypi/dm/hyperactive?style=for-the-badge&color=red)](https://pypi.org/project/hyperactive/)

```console
pip install hyperactive
```

<br>

## Example

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import fetch_california_housing
from hyperactive import Hyperactive

data = fetch_california_housing()
X, y = data.data, data.target

# define the model in a function
def model(opt):
    # pass the suggested parameter to the machine learning model
    gbr = GradientBoostingRegressor(
        n_estimators=opt["n_estimators"]
    )
    scores = cross_val_score(gbr, X, y, cv=3)

    # return a single numerical value, which gets maximized
    return scores.mean()


# search space determines the ranges of parameters you want the optimizer to search through
search_space = {"n_estimators": list(range(10, 200, 5))}

# start the optimization run
hyper = Hyperactive()
hyper.add_search(model, search_space, n_iter=50)
hyper.run()
```

<br>

## Hyperactive API reference


<br>

### Basic Usage

<details>
<summary><b> Hyperactive(verbosity, distribution, n_processes)</b></summary>

- verbosity = ["progress_bar", "print_results", "print_times"]
  - Possible parameter types: (list, False)
  - The verbosity list determines what part of the optimization information will be printed in the command line.

- distribution = "multiprocessing"
  - Possible parameter types: ("multiprocessing", "joblib", "pathos")
  - Determine, which distribution service you want to use. Each library uses different packages to pickle objects:
    - multiprocessing uses pickle
    - joblib uses dill
    - pathos uses cloudpickle
  
      
- n_processes = "auto",   
  - Possible parameter types: (str, int)
  - The maximum number of processes that are allowed to run simultaneously. If n_processes is of int-type there will only run n_processes-number of jobs simultaneously instead of all at once. So if n_processes=10 and n_jobs_total=35, then the schedule would look like this 10 - 10 - 10 - 5. This saves computational resources if there is a large number of n_jobs. If "auto", then n_processes is the sum of all n_jobs (from .add_search(...)).

</details>


<details>
<summary><b> .add_search(objective_function, search_space, n_iter, optimizer, n_jobs, initialize, pass_through, callbacks, catch, max_score, early_stopping, random_state, memory, memory_warm_start, progress_board)</b></summary>


- objective_function
  - Possible parameter types: (callable)
  - The objective function defines the optimization problem. The optimization algorithm will try to maximize the numerical value that is returned by the objective function by trying out different parameters from the search space.


- search_space
  - Possible parameter types: (dict)
  - Defines the space were the optimization algorithm can search for the best parameters for the given objective function.


- n_iter
  - Possible parameter types: (int)
  - The number of iterations that will be performed during the optimization run. The entire iteration consists of the optimization-step, which decides the next parameter that will be evaluated and the evaluation-step, which will run the objective function with the chosen parameter and return the score.


- optimizer = "default"
  - Possible parameter types: ("default", initialized optimizer object)
  - Instance of optimization class that can be imported from Hyperactive. "default" corresponds to the random search optimizer. The imported optimization classes from hyperactive are different from gfo. They only accept optimizer-specific-parameters. The following classes can be imported and used:
  
    - HillClimbingOptimizer
    - StochasticHillClimbingOptimizer
    - RepulsingHillClimbingOptimizer
    - SimulatedAnnealingOptimizer
    - DownhillSimplexOptimizer
    - RandomSearchOptimizer
    - GridSearchOptimizer
    - RandomRestartHillClimbingOptimizer
    - RandomAnnealingOptimizer
    - PowellsMethod
    - PatternSearch
    - ParallelTemperingOptimizer
    - ParticleSwarmOptimizer
    - EvolutionStrategyOptimizer
    - BayesianOptimizer
    - TreeStructuredParzenEstimators
    - ForestOptimizer
    
  - Example:
    ```python
    ...
    
    opt_hco = HillClimbingOptimizer(epsilon=0.08)
    hyper = Hyperactive()
    hyper.add_search(..., optimizer=opt_hco)
    hyper.run()
    
    ...
    ```


- n_jobs = 1
  - Possible parameter types: (int)
  - Number of jobs to run in parallel. Those jobs are optimization runs that work independent from another (no information sharing). If n_jobs == -1 the maximum available number of cpu cores is used.


- initialize = {"grid": 4, "random": 2, "vertices": 4}
  - Possible parameter types: (dict)
  - The initialization dictionary automatically determines a number of parameters that will be evaluated in the first n iterations (n is the sum of the values in initialize). The initialize keywords are the following:
    - grid
      - Initializes positions in a grid like pattern. Positions that cannot be put into a grid are randomly positioned. For very high dimensional search spaces (>30) this pattern becomes random.
    - vertices
      - Initializes positions at the vertices of the search space. Positions that cannot be put into a new vertex are randomly positioned.

    - random
      - Number of random initialized positions

    - warm_start
      - List of parameter dictionaries that marks additional start points for the optimization run.
  
    Example:
    ```python
    ... 
    search_space = {
        "x1": list(range(10, 150, 5)),
        "x2": list(range(2, 12)),
    }

    ws1 = {"x1": 10, "x2": 2}
    ws2 = {"x1": 15, "x2": 10}

    hyper = Hyperactive()
    hyper.add_search(
        model,
        search_space,
        n_iter=30,
        initialize={"grid": 4, "random": 10, "vertices": 4, "warm_start": [ws1, ws2]},
    )
    hyper.run()
    ```


- pass_through = {}
  - Possible parameter types: (dict)
  - The pass_through accepts a dictionary that contains information that will be passed to the objective-function argument. This information will not change during the optimization run, unless the user does so by himself (within the objective-function).
  
    Example:
    ```python
    ... 
    def objective_function(para):
        para.pass_through["stuff1"] # <--- this variable is 1
        para.pass_through["stuff2"] # <--- this variable is 2

        score = -para["x1"] * para["x1"]
        return score

    pass_through = {
      "stuff1": 1,
      "stuff2": 2,
    }

    hyper = Hyperactive()
    hyper.add_search(
        model,
        search_space,
        n_iter=30,
        pass_through=pass_through,
    )
    hyper.run()
    ```


- callbacks = {}
  - Possible parameter types: (dict)
  - The callbacks enables you to pass functions to hyperactive that are called every iteration during the optimization run. The function has access to the same argument as the objective-function. You can decide if the functions are called before or after the objective-function is evaluated via the keys of the callbacks-dictionary. The values of the dictionary are lists of the callback-functions. The following example should show they way to use callbacks: 


    Example:
    ```python
    ...

    def callback_1(access):
      # do some stuff

    def callback_2(access):
      # do some stuff

    def callback_3(access):
      # do some stuff

    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        n_iter=100,
        callbacks={
          "after": [callback_1, callback_2],
          "before": [callback_3]
          },
    )
    hyper.run()
    ```


- catch = {}
  - Possible parameter types: (dict)
  - The catch parameter provides a way to handle exceptions that occur during the evaluation of the objective-function or the callbacks. It is a dictionary that accepts the exception class as a key and the score that is returned instead as the value. This way you can handle multiple types of exceptions and return different scores for each. 
  In the case of an exception it often makes sense to return `np.nan` as a score. You can see an example of this in the following code-snippet:

    Example:
    ```python
    ...
    
    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        n_iter=100,
        catch={
          ValueError: np.nan,
          },
    )
    hyper.run()
    ```


- max_score = None
  - Possible parameter types: (float, None)
  - Maximum score until the optimization stops. The score will be checked after each completed iteration.


- early_stopping=None
  - (dict, None)
  - Stops the optimization run early if it did not achive any score-improvement within the last iterations. The early_stopping-parameter enables to set three parameters:
    - `n_iter_no_change`: Non-optional int-parameter. This marks the last n iterations to look for an improvement over the iterations that came before n. If the best score of the entire run is within those last n iterations the run will continue (until other stopping criteria are met), otherwise the run will stop.
    - `tol_abs`: Optional float-paramter. The score must have improved at least this absolute tolerance in the last n iterations over the best score in the iterations before n. This is an absolute value, so 0.1 means an imporvement of 0.8 -> 0.9 is acceptable but 0.81 -> 0.9 would stop the run.
    - `tol_rel`: Optional float-paramter. The score must have imporved at least this relative tolerance (in percentage) in the last n iterations over the best score in the iterations before n. This is a relative value, so 10 means an imporvement of 0.8 -> 0.88 is acceptable but 0.8 -> 0.87 would stop the run.

  - random_state = None
  - Possible parameter types: (int, None)
  - Random state for random processes in the random, numpy and scipy module.


- memory = True
  - Possible parameter types: (bool, "share")
  - Whether or not to use the "memory"-feature. The memory is a dictionary, which gets filled with parameters and scores during the optimization run. If the optimizer encounters a parameter that is already in the dictionary it just extracts the score instead of reevaluating the objective function (which can take a long time). If memory is set to "share" and there are multiple jobs for the same objective function then the memory dictionary is automatically shared between the different processes.

- memory_warm_start = None
  - Possible parameter types: (pandas dataframe, None)
  - Pandas dataframe that contains score and parameter information that will be automatically loaded into the memory-dictionary.

      example:

      <table class="table">
        <thead class="table-head">
          <tr class="row">
            <td class="cell">score</td>
            <td class="cell">x1</td>
            <td class="cell">x2</td>
            <td class="cell">x...</td>
          </tr>
        </thead>
        <tbody class="table-body">
          <tr class="row">
            <td class="cell">0.756</td>
            <td class="cell">0.1</td>
            <td class="cell">0.2</td>
            <td class="cell">...</td>
          </tr>
          <tr class="row">
            <td class="cell">0.823</td>
            <td class="cell">0.3</td>
            <td class="cell">0.1</td>
            <td class="cell">...</td>
          </tr>
          <tr class="row">
            <td class="cell">...</td>
            <td class="cell">...</td>
            <td class="cell">...</td>
            <td class="cell">...</td>
          </tr>
          <tr class="row">
            <td class="cell">...</td>
            <td class="cell">...</td>
            <td class="cell">...</td>
            <td class="cell">...</td>
          </tr>
        </tbody>
      </table>
  
  
- progress_board = None
  - Possible parameter types: (initialized ProgressBoard object, None)
  - Initialize the ProgressBoard class and pass the object to the progress_board-parameter. 


</details>



<details>
<summary><b> .run(max_time)</b></summary>

- max_time = None
  - Possible parameter types: (float, None)
  - Maximum number of seconds until the optimization stops. The time will be checked after each completed iteration.

</details>



<br>

### Special Parameters

<details>
<summary><b> Objective Function</b></summary>

Each iteration consists of two steps:
 - The optimization step: decides what position in the search space (parameter set) to evaluate next 
 - The evaluation step: calls the objective function, which returns the score for the given position in the search space
  
The objective function has one argument that is often called "para", "params", "opt" or "access".
This argument is your access to the parameter set that the optimizer has selected in the
corresponding iteration. 

```python
def objective_function(opt):
    # get x1 and x2 from the argument "opt"
    x1 = opt["x1"]
    x2 = opt["x1"]

    # calculate the score with the parameter set
    score = -(x1 * x1 + x2 * x2)

    # return the score
    return score
```

The objective function always needs a score, which shows how "good" or "bad" the current parameter set is. But you can also return some additional information with a dictionary:

```python
def objective_function(opt):
    x1 = opt["x1"]
    x2 = opt["x1"]

    score = -(x1 * x1 + x2 * x2)

    other_info = {
      "x1 squared" : x1**2,
      "x2 squared" : x2**2,
    }

    return score, other_info
```

When you take a look at the results (a pandas dataframe with all iteration information) after the run has ended you will see the additional information in it. The reason we need a dictionary for this is because Hyperactive needs to know the names of the additonal parameters. The score does not need that, because it is always called "score" in the results. You can run [this example script](https://github.com/SimonBlanke/Hyperactive/blob/master/examples/optimization_applications/multiple_scores.py) if you want to give it a try.

</details>


<details>
<summary><b> Search Space Dictionary</b></summary>

The search space defines what values the optimizer can select during the search. These selected values will be inside the objective function argument and can be accessed like in a dictionary. The values in each search space dimension should always be in a list. If you use np.arange you should put it in a list afterwards:

```python
search_space = {
    "x1": list(np.arange(-100, 101, 1)),
    "x2": list(np.arange(-100, 101, 1)),
}
```

A special feature of Hyperactive is shown in the next example. You can put not just numeric values into the search space dimensions, but also strings and functions. This enables a very high flexibility in how you can create your studies.

```python
def func1():
  # do stuff
  return stuff
  

def func2():
  # do stuff
  return stuff


search_space = {
    "x": list(np.arange(-100, 101, 1)),
    "str": ["a string", "another string"],
    "function" : [func1, func2],
}
```

If you want to put other types of variables (like numpy arrays, pandas dataframes, lists, ...) into the search space you can do that via functions:

```python
def array1():
  return np.array([1, 2, 3])
  

def array2():
  return np.array([3, 2, 1])


search_space = {
    "x": list(np.arange(-100, 101, 1)),
    "str": ["a string", "another string"],
    "numpy_array" : [array1, array2],
}
```

The functions contain the numpy arrays and returns them. This way you can use them inside the objective function. 


</details>


<details>
<summary><b> Optimizer Classes</b></summary>

Each of the following optimizer classes can be initialized and passed to the "add_search"-method via the "optimizer"-argument. During this initialization the optimizer class accepts **only optimizer-specific-paramters** (no random_state, initialize, ... ):
  
  ```python
  optimizer = HillClimbingOptimizer(epsilon=0.1, distribution="laplace", n_neighbours=4)
  ```
  
  for the default parameters you can just write:
  
  ```python
  optimizer = HillClimbingOptimizer()
  ```
  
  and pass it to Hyperactive:
  
  ```python
  hyper = Hyperactive()
  hyper.add_search(model, search_space, optimizer=optimizer, n_iter=100)
  hyper.run()
  ```
  
  So the optimizer-classes are **different** from Gradient-Free-Optimizers. A more detailed explanation of the optimization-algorithms and the optimizer-specific-paramters can be found in the [Optimization Tutorial](https://github.com/SimonBlanke/optimization-tutorial).

- HillClimbingOptimizer
- RepulsingHillClimbingOptimizer
- SimulatedAnnealingOptimizer
- DownhillSimplexOptimizer
- RandomSearchOptimizer
- GridSearchOptimizer
- RandomRestartHillClimbingOptimizer
- RandomAnnealingOptimizer
- PowellsMethod
- PatternSearch
- ParallelTemperingOptimizer
- ParticleSwarmOptimizer
- EvolutionStrategyOptimizer
- BayesianOptimizer
- TreeStructuredParzenEstimators
- ForestOptimizer

</details>



<br>

### Result Attributes


<details>
<summary><b> .best_para(objective_function)</b></summary>

- objective_function
  - (callable)
- returnes: dictionary
- Parameter dictionary of the best score of the given objective_function found in the previous optimization run.

  example:
  ```python
  {
    'x1': 0.2, 
    'x2': 0.3,
  }
  ```
  
</details>


<details>
<summary><b> .best_score(objective_function)</b></summary>

- objective_function
  - (callable)
- returns: int or float
- Numerical value of the best score of the given objective_function found in the previous optimization run.

</details>


<details>
<summary><b> .search_data(objective_function, times=False)</b></summary>

- objective_function
  - (callable)
- returns: Pandas dataframe 
- The dataframe contains score and parameter information of the given objective_function found in the optimization run. If the parameter `times` is set to True the evaluation- and iteration- times are added to the dataframe. 

    example:

    <table class="table">
      <thead class="table-head">
        <tr class="row">
          <td class="cell">score</td>
          <td class="cell">x1</td>
          <td class="cell">x2</td>
          <td class="cell">x...</td>
        </tr>
      </thead>
      <tbody class="table-body">
        <tr class="row">
          <td class="cell">0.756</td>
          <td class="cell">0.1</td>
          <td class="cell">0.2</td>
          <td class="cell">...</td>
        </tr>
        <tr class="row">
          <td class="cell">0.823</td>
          <td class="cell">0.3</td>
          <td class="cell">0.1</td>
          <td class="cell">...</td>
        </tr>
        <tr class="row">
          <td class="cell">...</td>
          <td class="cell">...</td>
          <td class="cell">...</td>
          <td class="cell">...</td>
        </tr>
        <tr class="row">
          <td class="cell">...</td>
          <td class="cell">...</td>
          <td class="cell">...</td>
          <td class="cell">...</td>
        </tr>
      </tbody>
    </table>

</details>



<br>

## Roadmap

<details>
<summary><b>v2.0.0</b> :heavy_check_mark:</summary>

  - [x] Change API
</details>

<details>
<summary><b>v2.1.0</b> :heavy_check_mark:</summary>

  - [x] Save memory of evaluations for later runs (long term memory)
  - [x] Warm start sequence based optimizers with long term memory
  - [x] Gaussian process regressors from various packages (gpy, sklearn, GPflow, ...) via wrapper
</details>

<details>
<summary><b>v2.2.0</b> :heavy_check_mark:</summary>

  - [x] Add basic dataset meta-features to long term memory
  - [x] Add helper-functions for memory
      - [x] connect two different model/dataset hashes
      - [x] split two different model/dataset hashes
      - [x] delete memory of model/dataset
      - [x] return best known model for dataset
      - [x] return search space for best model
      - [x] return best parameter for best model
</details>

<details>
<summary><b>v2.3.0</b> :heavy_check_mark:</summary>

  - [x] Tree-structured Parzen Estimator
  - [x] Decision Tree Optimizer
  - [x] add "max_sample_size" and "skip_retrain" parameter for sbom to decrease optimization time
</details>

<details>
<summary><b>v3.0.0</b> :heavy_check_mark:</summary>

  - [x] New API
      - [x] expand usage of objective-function
      - [x] No passing of training data into Hyperactive
      - [x] Removing "long term memory"-support (better to do in separate package)
      - [x] More intuitive selection of optimization strategies and parameters
      - [x] Separate optimization algorithms into other package
      - [x] expand api so that optimizer parameter can be changed at runtime
      - [x] add extensive testing procedure (similar to Gradient-Free-Optimizers)

</details>

<details>
<summary><b>v3.1.0</b> :heavy_check_mark:</summary>

  - [x] Decouple number of runs from active processes (Thanks to [PartiallyTyped](https://github.com/PartiallyTyped))

</details>

<details>
<summary><b>v3.2.0</b> :heavy_check_mark:</summary>

  - [x] Dashboard for visualization of search-data at runtime via streamlit (Progress-Board)

</details>

<details>
<summary><b>v3.3.0</b> :heavy_check_mark:</summary>

  - [x] Early stopping 
  - [x] Shared memory dictionary between processes with the same objective function

</details>

<details>
<summary><b>v4.0.0</b> :heavy_check_mark:</summary>

  - [x] small adjustments to API
  - [x] move optimization strategies into sub-module "optimizers"
  - [x] preparation for future add ons (long-term-memory, meta-learn, ...) from separate repositories
  - [x] separate progress board into separate repository

</details>

<details>
<summary><b>v4.1.0</b> :heavy_check_mark:</summary>

  - [x] add python 3.9 to testing
  - [x] add pass_through-parameter
  - [x] add v1 GFO optimization algorithms

</details>

<details>
<summary><b>v4.2.0</b> </summary>

  - [x] add callbacks-parameter
  - [x] add catch-parameter
  - [x] add option to add eval- and iter- times to search-data

</details>


<details>
<summary><b>Upcoming Features</b></summary>
   
  - [ ] Data collector tool to store data (from inside the objective function) into csv-files
  - [ ] Experiment-tracking for search-data storage and usage
  - [ ] Dashboard for visualization of stored search-data
  - [ ] Meta-Learning tool for hyperparameter optimization

</details>





<br>


## Experimental algorithms

The following algorithms are of my own design and, to my knowledge, do not yet exist in the technical literature.
If any of these algorithms already exist I would like you to share it with me in an issue.

#### Random Annealing

A combination between simulated annealing and random search.


<br>

## FAQ

#### Known Errors + Solutions

<details>
<summary><b> Read this before opening a bug-issue </b></summary>

<br>
  
- <b>Are you sure the bug is located in Hyperactive? </b>

  The error might be located in the optimization-backend. 
  Look at the error message from the command line. <b>If</b> one of the last messages look like this:
     - File "/.../gradient_free_optimizers/...", line ...

  <b>Then</b> you should post the bug report in: 
     - https://github.com/SimonBlanke/Gradient-Free-Optimizers

  <br>Otherwise</b> you can post the bug report in Hyperactive
  
- <b>Do you have the correct Hyperactive version? </b>
  
  Every major version update (e.g. v2.2 -> v3.0) the API of Hyperactive changes.
  Check which version of Hyperactive you have. If your major version is older you have two options:
  
  <b>Recommended:</b> You could just update your Hyperactive version with:
  ```bash
  pip install hyperactive --upgrade
  ```
  This way you can use all the new documentation and examples from the current repository.
    
  Or you could continue using the old version and use an old repository branch as documentation.
  You can do that by selecting the corresponding branch. (top right of the repository. The default is "master" or "main")
  So if your major version is older (e.g. v2.1.0) you can select the 2.x.x branch to get the old repository for that version.
  
- <b>Provide example code for error reproduction </b>
  To understand and fix the issue I need an example code to reproduce the error.
  I must be able to just copy the code into a py-file and execute it to reproduce the error.
  
</details>


<details>
<summary> MemoryError: Unable to allocate ... for an array with shape (...) </summary>

<br>

This is expected of the current implementation of smb-optimizers. For all Sequential model based algorithms you have to keep your eyes on the search space size:
```python
search_space_size = 1
for value_ in search_space.values():
    search_space_size *= len(value_)
    
print("search_space_size", search_space_size)
```
Reduce the search space size to resolve this error.

</details>


<details>
<summary> TypeError: cannot pickle '_thread.RLock' object </summary>

<br>

This is because you have classes and/or non-top-level objects in the search space. Pickle (used by multiprocessing) cannot serialize them. Setting distribution to "joblib" or "pathos" may fix this problem:
```python
hyper = Hyperactive(distribution="joblib")
```

</details>


<details>
<summary> Command line full of warnings </summary>

<br>

Very often warnings from sklearn or numpy. Those warnings do not correlate with bad performance from Hyperactive. Your code will most likely run fine. Those warnings are very difficult to silence.

It should help to put this at the very top of your script:
```python
def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn
```

</details>


<details>
<summary> Warning: Not enough initial positions for population size </summary>

<br>
  
This warning occurs because Hyperactive needs more initial positions to choose from to generate a population for the optimization algorithm:
The number of initial positions is determined by the `initialize`-parameter in the `add_search`-method.
```python
# This is how it looks per default
initialize = {"grid": 4, "random": 2, "vertices": 4}
  
# You could set it to this for a maximum population of 20
initialize = {"grid": 4, "random": 12, "vertices": 4}
```
  
</details>



<br>

## References

#### [dto] [Scikit-Optimize](https://github.com/scikit-optimize/scikit-optimize/blob/master/skopt/learning/forest.py)

<br>

## Citing Hyperactive

    @Misc{hyperactive2021,
      author =   {{Simon Blanke}},
      title =    {{Hyperactive}: An optimization and data collection toolbox for convenient and fast prototyping of computationally expensive models.},
      howpublished = {\url{https://github.com/SimonBlanke}},
      year = {since 2019}
    }


<br>

## License

[![LICENSE](https://img.shields.io/github/license/SimonBlanke/Hyperactive?style=for-the-badge)](https://github.com/SimonBlanke/Hyperactive/blob/master/LICENSE)
