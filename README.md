<p align="center">
  <a href="https://github.com/SimonBlanke/Hyperactive"><img src="./docs/images/logo.png" height="240"></a>
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

- visualizes your collected data (> v3.1.0)

- saves your [computation time](./examples/optimization_applications/memory.py)

- supports [parallel computing](./examples/tested_and_supported_packages/multiprocessing_example.py)


<br>
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
  - ### [Hyperactive Tutorial](https://nbviewer.jupyter.org/github/SimonBlanke/hyperactive-tutorial/blob/main/notebooks/hyperactive_tutorial.ipynb)
  - ### [Neural Architecture Search Tutorial](https://nbviewer.jupyter.org/github/SimonBlanke/hyperactive-tutorial/blob/main/notebooks/Optimization%20Strategies%20for%20Deep%20Learning.ipynb)


<br>

## Overview

<h3 align="center">
Hyperactive features a collection of optimization algorithms that can be used for a variety of optimization problems. The following table shows listings of the capabilities of Hyperactive, where each of the items links to an example:
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
         </ul><br>
        <a><b>Global Search:</b></a>
          <ul>
            <li><a href="./examples/optimization_techniques/random_search.py">Random Search</a></li>
            <li><a href="./examples/optimization_techniques/rand_rest_hill_climbing.py">Random Restart Hill Climbing</a></li>
            <li><a href="./examples/optimization_techniques/random_annealing.py">Random Annealing</a> [<a href="#/./overview#experimental-algorithms">*</a>] </li>
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
            <li><a href="./examples/optimization_techniques/decision_tree_optimization.py">Decision Tree Optimizer</a>
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
            <li><a href="./examples/optimization_applications/meta_data_collection.py">Meta-data Collection</a></li>
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
<tr>
<th> Regular training </th>
<th> Hyperactive </th>
</tr>
<tr>
<td>
<sub>

```python
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_boston


data = load_boston()
X, y = data.data, data.target


gbr = DecisionTreeRegressor(max_depth=10)
score = cross_val_score(gbr, X, y, cv=3).mean()










```

</sub>
</td>
<td>
<sup>

```python
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_boston
from hyperactive import Hyperactive

data = load_boston()
X, y = data.data, data.target

def model(opt):
    gbr = DecisionTreeRegressor(max_depth=opt["max_depth"])
    return cross_val_score(gbr, X, y, cv=3).mean()


search_space = {"max_depth": list(range(3, 25))}

hyper = Hyperactive()
hyper.add_search(model, search_space, n_iter=50)
hyper.run()
```

</sub>
</td>
</tr>
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
from sklearn.datasets import load_boston
from hyperactive import Hyperactive

data = load_boston()
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
  - (list, False)
  - The verbosity list determines what part of the optimization information will be printed in the command line.

- distribution = {"multiprocessing": {"initializer": tqdm.set_lock, "initargs": (tqdm.get_lock(),),}}
  - (str, dict, callable)
  - Access the parallel processing in three ways:
    - Via a str "multiprocessing" or "joblib" to choose one of the two.
    - Via a dictionary with one key "multiprocessing" or "joblib" and a value that is the input argument of Pool and Parallel. The default argument is a good example of this.
    - Via your own parallel processing function that will be used instead of those for multiprocessing and joblib. The wrapper-function must work similar to the following two functions:
    
    Multiprocessing:
    ```python
    def multiprocessing_wrapper(process_func, search_processes_paras, **kwargs):
      n_jobs = len(search_processes_paras)

      pool = Pool(n_jobs, **kwargs)
      results = pool.map(process_func, search_processes_paras)

      return results
    ```
    
    Joblib:
    ```python
    def joblib_wrapper(process_func, search_processes_paras, **kwargs):
        n_jobs = len(search_processes_paras)

        jobs = [
            delayed(process_func)(**info_dict)
            for info_dict in search_processes_paras
        ]
        results = Parallel(n_jobs=n_jobs, **kwargs)(jobs)

        return results
      ```
      
- n_processes = "auto",   
  - (str, int)
  - The maximum number of processes that are allowed to run simultaneously. If n_processes is of int-type there will only run n_processes-number of jobs simultaneously instead of all at once. So if n_processes=10 and n_jobs_total=35, then the schedule would look like this 10 - 10 - 10 - 5. This saves computational resources if there is a large number of n_jobs. If "auto", then n_processes is the sum of all n_jobs (from .add_search(...)).

</details>


<details>
<summary><b> .add_search(objective_function, search_space, n_iter, optimizer, n_jobs, initialize, max_score, random_state, memory, memory_warm_start)</b></summary>


- objective_function
  - (callable)
  - The objective function defines the optimization problem. The optimization algorithm will try to maximize the numerical value that is returned by the objective function by trying out different parameters from the search space.

- search_space
  - (dict)
  - Defines the space were the optimization algorithm can search for the best parameters for the given objective function.

- n_iter
  - (int)
  - The number of iterations that will be performed during the optimization run. The entire iteration consists of the optimization-step, which decides the next parameter that will be evaluated and the evaluation-step, which will run the objective function with the chosen parameter and return the score.

- optimizer = "default"
  - (object)
  - Instance of optimization class that can be imported from Hyperactive. "default" corresponds to the random search optimizer. The following classes can be imported and used:
  
    - HillClimbingOptimizer
    - StochasticHillClimbingOptimizer
    - RepulsingHillClimbingOptimizer
    - RandomSearchOptimizer
    - RandomRestartHillClimbingOptimizer
    - RandomAnnealingOptimizer
    - SimulatedAnnealingOptimizer
    - ParallelTemperingOptimizer
    - ParticleSwarmOptimizer
    - EvolutionStrategyOptimizer
    - BayesianOptimizer
    - TreeStructuredParzenEstimators
    - DecisionTreeOptimizer
    - EnsembleOptimizer
    
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
  - (int)
  - Number of jobs to run in parallel. Those jobs are optimization runs that work independent from another (no information sharing). If n_jobs == -1 the maximum available number of cpu cores is used.

- initialize = {"grid": 4, "random": 2, "vertices": 4}
  - (dict)
  - The initialization dictionary automatically determines a number of parameters that will be evaluated in the first n iterations (n is the sum of the values in initialize). The initialize keywords are the following:
    - grid
      - Initializes positions in a grid like pattern. Positions that cannot be put into a grid are randomly positioned.
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
  
  
  
- max_score = None
  - (float, None)
  - Maximum score until the optimization stops. The score will be checked after each completed iteration.

- random_state = None
  - (int, None)
  - Random state for random processes in the random, numpy and scipy module.

- memory = True
  - (bool)
  - Whether or not to use the "memory"-feature. The memory is a dictionary, which gets filled with parameters and scores during the optimization run. If the optimizer encounters a parameter that is already in the dictionary it just extracts the score instead of reevaluating the objective function (which can take a long time).

- memory_warm_start = None
  - (pandas dataframe, None)
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
  
</details>



<details>
<summary><b> .run(max_time)</b></summary>

- max_time = None
  - (float, None)
  - Maximum number of seconds until the optimization stops. The time will be checked after each completed iteration.

</details>



<br>

### Special Parameters

<details>
<summary><b> Objective Function</b></summary>

Each iteration consists of two steps:
 - The optimization step: decides what position in the search space (parameter set) to evaluate next 
 - The evaluation step: calls the objective function, which returns the score for the given position in the search space
  
The objective function has one argument that is often called "para", "params" or "opt".
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
  return np.array([0, 1, 2])
  

def array2():
  return np.array([0, 1, 2])


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

Each of the following optimizer classes can be initialized and passed to the "add_search"-method via the "optimizer"-argument. During this initialization the optimizer class accepts additional paramters.
You can read more about each optimization-strategy and its parameters in the [Optimization Tutorial](https://github.com/SimonBlanke/optimization-tutorial).

- HillClimbingOptimizer
- RepulsingHillClimbingOptimizer
- SimulatedAnnealingOptimizer
- RandomSearchOptimizer
- RandomRestartHillClimbingOptimizer
- RandomAnnealingOptimizer
- ParallelTemperingOptimizer
- ParticleSwarmOptimizer
- EvolutionStrategyOptimizer
- BayesianOptimizer
- TreeStructuredParzenEstimators
- DecisionTreeOptimizer

</details>


</details>


<details>
<summary><b> Progress Board</b></summary>

The progress board enables the visualization of search data during the optimization run. This will help you to understand what is happening during the optimization and give an overview of the explored parameter sets and scores. 


- filter_file
  - (None, True)
  - If the filter_file-parameter is True Hyperactive will create a file in the current directory, which allows the filtering of parameters or the score by setting an upper or lower bound.


The following script provides an example:

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import load_boston

from hyperactive import Hyperactive
# import the ProgressBoard
from hyperactive.dashboards import ProgressBoard

data = load_boston()
X, y = data.data, data.target


def model(opt):
    gbr = GradientBoostingRegressor(
        n_estimators=opt["n_estimators"],
        max_depth=opt["max_depth"],
        min_samples_split=opt["min_samples_split"],
    )
    scores = cross_val_score(gbr, X, y, cv=3)

    return scores.mean()


search_space = {
    "n_estimators": list(range(50, 150, 5)),
    "max_depth": list(range(2, 12)),
    "min_samples_split": list(range(2, 22)),
}

# create an instance of the ProgressBoard
progress_board = ProgressBoard()

hyper = Hyperactive()

# pass the instance of the ProgressBoard to .add_search(...)
hyper.add_search(
    model,
    search_space,
    n_iter=120,
    progress_board=progress_board,
)

# a terminal will open, which opens a dashboard in your browser
hyper.run()
```

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
<summary><b> .results(objective_function)</b></summary>

- objective_function
  - (callable)
- returns: Pandas dataframe 
- The dataframe contains score, parameter information, iteration times and evaluation times of the given objective_function found in the previous optimization run.

    example:

    <table class="table">
      <thead class="table-head">
        <tr class="row">
          <td class="cell">score</td>
          <td class="cell">x1</td>
          <td class="cell">x2</td>
          <td class="cell">x...</td>
          <td class="cell">eval_times</td>
          <td class="cell">iter_times</td>
        </tr>
      </thead>
      <tbody class="table-body">
        <tr class="row">
          <td class="cell">0.756</td>
          <td class="cell">0.1</td>
          <td class="cell">0.2</td>
          <td class="cell">...</td>
          <td class="cell">0.953</td>
          <td class="cell">1.123</td>
        </tr>
        <tr class="row">
          <td class="cell">0.823</td>
          <td class="cell">0.3</td>
          <td class="cell">0.1</td>
          <td class="cell">...</td>
          <td class="cell">0.948</td>
          <td class="cell">1.101</td>
        </tr>
        <tr class="row">
          <td class="cell">...</td>
          <td class="cell">...</td>
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
<summary><b>Next Features</b></summary>
  
  - [ ] "long term memory" for search-data storage and usage
  - [ ] Data collector tool to use inside the objective function
  - [ ] Dashboard for visualization of stored search-data


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

Are you sure the bug is located in Hyperactive?

Look at the error message from the command line. <b>If</b> one of the last messages look like this:
   - File "/.../gradient_free_optimizers/...", line ...

<b>Then</b> you should post the bug report in: 
   - https://github.com/SimonBlanke/Gradient-Free-Optimizers

<b>Otherwise</b> you can post the bug report in Hyperactive

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

Setting distribution to "joblib" may fix this problem:
```python
hyper = Hyperactive(distribution="joblib")
```

</details>


<details>
<summary> Command line full of warnings </summary>

<br>

Very often warnings from sklearn or numpy. Those warnings do not correlate with bad performance from Hyperactive. Your code will most likely run fine. Those warnings are very difficult to silence.

Put this at the very top of your script:
```python
def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn
```

</details>




<br>

## References

#### [dto] [Scikit-Optimize](https://github.com/scikit-optimize/scikit-optimize/blob/master/skopt/learning/forest.py)

<br>

## Citing Hyperactive

    @Misc{hyperactive2019,
      author =   {{Simon Blanke}},
      title =    {{Hyperactive}: An optimization and data collection toolbox for convenient and fast prototyping of computationally expensive models.},
      howpublished = {\url{https://github.com/SimonBlanke}},
      year = {since 2019}
    }


<br>

## License

[![LICENSE](https://img.shields.io/github/license/SimonBlanke/Hyperactive?style=for-the-badge)](https://github.com/SimonBlanke/Hyperactive/blob/master/LICENSE)
