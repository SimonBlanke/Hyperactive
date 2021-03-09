<p align="center">
  <a href="https://simonblanke.github.io/Hyperactive/"><img src="./docs/images/logo.png" height="240"></a>
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

<h2 align="center">A hyperparameter optimization and meta-learning toolbox for convenient and fast prototyping of machine-learning models.</h2>

<br>

<img src="./docs/images/bayes_ackley.gif" align="right" alt="logo" width="500" height="400">

## Hyperactive:

- is very easy to learn but extremly versatile

- provides intelligent optimization algorithms

- makes optimization data collection simple

- visualizes your collected data

- saves your computation time

- supports parallel computing


<br>
<br>
<br>
<br>
<br>

---

<div align="center"><a name="menu"></a>
  <h3>
    <a href="https://github.com/SimonBlanke/Hyperactive#main-features">Main features</a> •
    <a href="https://github.com/SimonBlanke/Hyperactive#installation">Installation</a> •
    <a href="https://github.com/SimonBlanke/Hyperactive#hyperactive-api-reference">API reference</a> •
    <a href="https://github.com/SimonBlanke/Hyperactive#roadmap">Roadmap</a> •
    <a href="https://github.com/SimonBlanke/Hyperactive#citing-hyperactive">Citation</a> •
    <a href="https://github.com/SimonBlanke/Hyperactive#license">License</a>
  </h3>
</div>

---


<br>

Hyperactive is primarily a <b>hyperparameter optimization toolkit</b>, that aims to simplify the model-selection and -tuning process. You can use any machine- or deep-learning package and it is not necessary to learn new syntax. Hyperactive offers <b>high versatility</b> in model optimization because of two characteristics:

  - You can define any kind of model in the objective function. It just has to return a score/metric that gets maximized.
  - The search space accepts not just 'int', 'float' or 'str' as data types but even functions, classes or any python objects.



<br>



<div align="center"></a>
<h3>
Hyperactive features a collection of optimization algorithms that can be used for a variety of optimization problems. The following table shows listings of the capabilities of Hyperactive, where each of the items links to an example:
</h3>
</div>

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
            <li><a href="https://simonblanke.github.io/Hyperactive/#/./optimizers/sequential_me./examples/optimization_techniques/tpe.py">Tree of Parzen Estimators</a></li>
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
        <a><b>Meta-data:</b></a>
          <ul>
            <li><a href="./examples/optimization_applications/meta_data_collection.py">Meta-data Collection</a></li>
            <li><a href="./examples/optimization_applications/meta_optimization.py">Meta Optimization</a></li>
            <li><a href="./examples/optimization_applications/meta_learning.py">Meta Learning</a></li>
          </ul>
        <a><b>Miscellaneous:</b></a>
          <ul>
            <li><a href="./examples/optimization_applications/test_function.py">Test Functions</a></li>
            <li>Fit Gaussian Curves</li>
            <li><a href="./examples/optimization_applications/multiple_scores.py">Managing multiple objectives</a></li>
          </ul>
      </td>
    </tr>
  </tbody>
</table>

The examples above are not necessarily done with realistic datasets or training procedures. 
The purpose is fast execution of the solution proposal and giving the user ideas for interesting usecases.



<br>

## Tutorials

  - [Optimization Strategies for Deep Learning with Hyperactive](https://nbviewer.jupyter.org/github/SimonBlanke/hyperactive-tutorial/blob/main/notebooks/Optimization%20Strategies%20for%20Deep%20Learning.ipynb)


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

<details>
<summary><b> Hyperactive(verbosity, distribution)</b></summary>

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
      - Initializes positions at the vertices of the search space. Positions that cannot be put into a vertices are randomly positioned.

    - random
      - Number of random initialized positions

    - warm_start
      - List of parameter dictionaries that marks additional start points for the optimization run.

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

### Optimizer Classes

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

<details open>
<summary><b>v3.1.0</b></summary>

  - [ ] New implementation of dashboard for visualization of search-data


</details>

<details>
<summary><b>v3.2.0</b></summary>

  - [ ] New implementation of "long term memory" for search-data storage and usage


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
      title =    {{Hyperactive}: A hyperparameter optimization and meta-learning toolbox for machine-/deep-learning models.},
      howpublished = {\url{https://github.com/SimonBlanke}},
      year = {since 2019}
    }


<br>

## License

[![LICENSE](https://img.shields.io/github/license/SimonBlanke/Hyperactive?style=for-the-badge)](https://github.com/SimonBlanke/Hyperactive/blob/master/LICENSE)
