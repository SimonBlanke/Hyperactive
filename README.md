<p align="center">
  <br>
  <a href="https://simonblanke.github.io/Hyperactive/"><img src="./docs/_media/hyperactive_logo.png" height="200"></a>
  <br>
</p>

<br>

---

<h2 align="center">A hyperparameter optimization and meta-learning toolbox for convenient and fast prototyping of machine-learning models.</h2>

<br>

<table>
  <tbody>
    <tr align="left" valign="center">
      <td>
        <strong>Master status:</strong>
      </td>
      <td>
        <a href="https://travis-ci.com/SimonBlanke/Hyperactive">
          <img src="https://img.shields.io/travis/com/SimonBlanke/Hyperactive/master?style=flat-square&logo=travis" alt="img not loaded: try F5 :)">
        </a>
        <a href="https://coveralls.io/github/SimonBlanke/Hyperactive">
          <img src="https://img.shields.io/coveralls/github/SimonBlanke/Hyperactive?style=flat-square&logo=codecov" alt="img not loaded: try F5 :)">
        </a>
      </td>
    </tr>
    <tr/>
    <tr align="left" valign="center">
      <td>
        <strong>Dev status:</strong>
      </td>
      <td>
        <a href="https://travis-ci.com/SimonBlanke/Hyperactive">
          <img src="https://img.shields.io/travis/SimonBlanke/Hyperactive/dev?style=flat-square&logo=travis" alt="img not loaded: try F5 :)">
        </a>
        <a href="https://coveralls.io/github/SimonBlanke/Hyperactive?branch=dev">
          <img src="https://img.shields.io/coveralls/github/SimonBlanke/Hyperactive/dev?style=flat-square&logo=codecov" alt="img not loaded: try F5 :)">
        </a>
      </td>
    </tr>
    <tr/>    <tr align="left" valign="center">
      <td>
         <strong>Code quality:</strong>
      </td>
      <td>
        <a href="https://codeclimate.com/github/SimonBlanke/Hyperactive">
        <img src="https://img.shields.io/codeclimate/maintainability/SimonBlanke/Hyperactive?style=flat-square&logo=code-climate" alt="img not loaded: try F5 :)">
        </a>
        <a href="https://scrutinizer-ci.com/g/SimonBlanke/Hyperactive/">
        <img src="https://img.shields.io/scrutinizer/quality/g/SimonBlanke/Hyperactive?style=flat-square&logo=scrutinizer-ci" alt="img not loaded: try F5 :)">
        </a>
      </td>
    </tr>
    <tr/>    <tr align="left" valign="center">
      <td>
        <strong>Latest versions:</strong>
      </td>
      <td>
        <a href="https://github.com/SimonBlanke/Hyperactive/releases">
          <img src="https://img.shields.io/github/v/release/SimonBlanke/Hyperactive?style=flat-square&logo=github" alt="img not loaded: try F5 :)">
        </a>
        <a href="https://pypi.org/project/hyperactive/">
          <img src="https://img.shields.io/pypi/v/Hyperactive?style=flat-square&logo=PyPi&logoColor=white" alt="img not loaded: try F5 :)">
        </a>
      </td>
    </tr>
  </tbody>
</table>

<br>

NEWS:

Hyperactive is currently in a transition phase between version 2 and 3. All of the source code for the optimization algorithms will be stored in <a href="https://github.com/SimonBlanke/Gradient-Free-Optimizers">Gradient-Free-Optimizers</a>. Gradient-Free-Optimizers will serve as an easy to use stand alone package as well as the optimization backend for Hyperactive in the future. Until Hyperactive version 3 is released you can either switch to the 2.x.x branch for the old version in Github or use Gradient-Free-Optimizers to enjoy new algorithms and improved performance.



<br>

Hyperactive is primarly a <b>hyperparameter optimization toolkit</b>, that aims to simplify the model-selection and -tuning process. You can use any machine- or deep-learning package and it is not necessary to learn new syntax. Hyperactive offers <b>high versatility</b> in model optimization because of two characteristics:

  - You can define any kind of model in the objective function. It just has to return a score/metric that gets maximized.
  - The search space accepts not just 'int', 'float' or 'str' as data types but even functions, classes or any python objects.


<br>

---

<div align="center"><a name="menu"></a>
  <h3>
    <a href="https://github.com/SimonBlanke/Hyperactive#main-features">Main features</a> •
    <a href="https://github.com/SimonBlanke/Hyperactive#installation">Installation</a> •
    <a href="https://github.com/SimonBlanke/Hyperactive#roadmap">Roadmap</a> •
    <a href="https://github.com/SimonBlanke/Hyperactive#citing-hyperactive">Citation</a> •
    <a href="https://github.com/SimonBlanke/Hyperactive#license">License</a>
  </h3>
</div>

---

<br>

## Main features

  - Thoroughly tested code base
  - Compatible with <b>any python machine-learning framework</b>
  - Optimize:
    - Anything from [simple models](https://simonblanke.github.io/Hyperactive/#/./examples/machine_learning?id=sklearn) <br/> to complex [machine-learning-pipelines](https://simonblanke.github.io/Hyperactive/#/./examples/use_cases?id=sklearn-pipeline)
    - Multi-level [ensembles](https://simonblanke.github.io/Hyperactive/#/./examples/use_cases?id=stacking)
    - [Deep neural network](https://simonblanke.github.io/Hyperactive/#/./examples/use_cases?id=neural-architecture-search) architecture
    - Other [optimization techniques](https://simonblanke.github.io/Hyperactive/#/./examples/use_cases?id=meta-optimization) (meta-optimization)
    - Or [any function](https://simonblanke.github.io/Hyperactive/#/./examples/test_functions?id=rosenbrock-function) you can specify with this API
  - Utilize state of the art optimization techniques like:
    - Simulated annealing
    - Evolution strategy
    - Bayesian optimization
  - [High performance](https://simonblanke.github.io/Hyperactive/#/./performance?id=performance): Optimizer time is neglectable for most models
  - Choose from a variety of different [optimization extensions](https://simonblanke.github.io/Hyperactive/#/./examples/extensions) to improve the optimization

<br>

<table>
  <tbody>
    <tr align="center" valign="center">
      <td>
        <strong>Optimization Techniques</strong>
        <img src="./docs/_media/blue.jpg"/>
      </td>
      <td>
        <strong>Tested and Supported Packages</strong>
        <img src="./docs/_media/blue.jpg"/>
      </td>
      <td>
        <strong>Optimization Applications</strong>
        <img src="./docs/_media/blue.jpg"/>
      </td>
    </tr>
    <tr/>
    <tr valign="top">
      <td>
        <a><b>Local Search:</b></a>
          <ul>
            <li><a href="https://simonblanke.github.io/Hyperactive#/./optimizers/local_search?id=Hill-Climbing">Hill Climbing</a></li>
            <li><a href="https://simonblanke.github.io/Hyperactive#/./optimizers/local_search?id=stochastic-hill-climbing">Stochastic Hill Climbing</a></li>
            <li><a href="https://simonblanke.github.io/Hyperactive#/./optimizers/local_search?id=tabu-search">Tabu Search</a></li>
            <li><a href="https://simonblanke.github.io/Hyperactive#/./optimizers/local_search?id=simulated-annealing">Simulated Annealing</a></li>
         </ul><br>
        <a><b>Global Search:</b></a>
          <ul>
            <li><a href="https://simonblanke.github.io/Hyperactive#/./optimizers/random_methods?id=random-search">Random Search</a></li>
            <li><a href="https://simonblanke.github.io/Hyperactive#/./optimizers/random_methods?id=random-restart-hill-climbing">Random Restart Hill Climbing</a></li>
            <li><a href="https://simonblanke.github.io/Hyperactive#/./optimizers/random_methods?id=random-annealing">Random Annealing</a> [<a href="#/./overview#experimental-algorithms">*</a>] </li>
         </ul><br>
        <a><b>Population Methods:</b></a>
          <ul>
            <li><a href="https://simonblanke.github.io/Hyperactive#/./optimizers/population_methods?id=parallel-tempering">Parallel Tempering</a></li>
            <li><a href="https://simonblanke.github.io/Hyperactive#/./optimizers/population_methods?id=particle-swarm-optimization">Particle Swarm Optimizer</li>
            <li><a href="https://simonblanke.github.io/Hyperactive#/./optimizers/population_methods?id=evolution-strategy">Evolution Strategy</a></li>
          </ul><br>
        <a><b>Sequential Methods:</b></a>
          <ul>
            <li><a href="https://simonblanke.github.io/Hyperactive#/./optimizers/sequential_methods?id=bayesian-optimization">Bayesian Optimization</a></li>
            <li><a href="https://simonblanke.github.io/Hyperactive/#/./optimizers/sequential_methods?id=tree-of-parzen-estimators">Tree of Parzen Estimators</a></li>
            <li><a href="https://simonblanke.github.io/Hyperactive/#/./optimizers/sequential_methods?id=decision-tree-optimizer">Decision Tree Optimizer</a>
            [<a href="#/./overview#references">dto</a>] </li>
          </ul>
      </td>
      <td>
        <a><b>Machine Learning:</b></a>
          <ul>
              <li><a href="https://simonblanke.github.io/Hyperactive#/./examples/machine_learning?id=sklearn">Scikit-learn</a></li>
              <li><a href="https://simonblanke.github.io/Hyperactive#/./examples/machine_learning?id=xgboost">XGBoost</a></li>
              <li><a href="https://simonblanke.github.io/Hyperactive#/./examples/machine_learning?id=lightgbm">LightGBM</a></li>
              <li><a href="https://simonblanke.github.io/Hyperactive#/./examples/machine_learning?id=catboost">CatBoost</a></li>
              <li><a href="https://simonblanke.github.io/Hyperactive#/./examples/machine_learning?id=rgf">RGF</a></li>
              <li><a href="https://simonblanke.github.io/Hyperactive#/./examples/machine_learning?id=mlxtend">Mlxtend</a></li>
          </ul><br>
        <a><b>Deep Learning:</b></a>
          <ul>
              <li><a href="https://simonblanke.github.io/Hyperactive#/./examples/deep_learning?id=tensorflow">Tensorflow</a></li>
              <li><a href="https://simonblanke.github.io/Hyperactive#/./examples/deep_learning?id=keras-cnn">Keras</a></li>
              <li>Pytorch</li>
          </ul><br>
        <a><b>Distribution:</b></a>
          <ul>
              <li><a href="https://simonblanke.github.io/Hyperactive#/./examples/distribution?id=multiprocessing">Multiprocessing</a></li>
              <li>Joblib</li>
          </ul>
      </td>
      <td>
        <a><b>Feature Engineering:</b></a>
          <ul>
            <li>Feature Transformation</li>
            <li>Feature Selection</li>
            <li>Feature Construction</li>
          </ul>
        <a><b>Machine Learning:</b></a>
          <ul>
            <li>Hyperparameter Tuning</li>
            <li>Model Selection</li>
            <li>Sklearn Pipelines</li>
            <li>Ensemble Learning</li>
          </ul>
        <a><b>Deep Learning:</b></a>
          <ul>
            <li>Neural Architecture Search</li>
            <li>Efficient Neural Architecture Search</li>
            <li>Transfer Learning</li>
          </ul>
        <a><b>Meta-data:</b></a>
          <ul>
            <li>Meta-data Collection</li>
            <li>Meta Optimization</li>
            <li>Meta Learning</li>
          </ul>
        <a><b>Miscellaneous:</b></a>
          <ul>
            <li>Test Functions</li>
            <li>Fit Gaussian Curves</li>
          </ul>
      </td>
    </tr>
  </tbody>
</table>

<br>

## Installation
[![PyPI version](https://badge.fury.io/py/hyperactive.svg)](https://badge.fury.io/py/hyperactive)

The most recent version of Hyperactive is available on PyPi:
```console
pip install hyperactive
```

<br>

## Minimal example

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import load_boston
from hyperactive import Hyperactive

data = load_boston()
X, y = data.data, data.target

""" define the model in a function """
def model(optimizer):
    """ pass the suggested parameter to the machine learning model """
    gbr = GradientBoostingRegressor(
        n_estimators=optimizer.suggested_params["n_estimators"]
    )
    scores = cross_val_score(gbr, X, y, cv=3)

    """ return a single numerical value, which gets maximized """
    return scores.mean()


""" 
create the search space 
determines the ranges of parameters you want the optimizer to search through
"""
search_space = {"n_estimators": list(range(10, 200, 5))}

""" start the optimization run """
hyper = Hyperactive()
hyper.search(model, search_space, n_iter=50)
```


## Hyperactive API information

<details>
<summary><b> Hyperactive(...)</b></summary>

    - random_state
    - verbosity

</details>


<details>
<summary><b> .search(...)</b></summary>

    - model
    - search_space
    - n_iter
    - optimizer=RandomSearchOptimizer()
    - max_time=None
    - n_jobs=1
    - initialize={"grid": 4, "random": 2, "vertices": 4}
    - memory=True

</details>

### Optimizers

<details>
<summary><b> HillClimbingOptimizer</b></summary>

    - epsilon=0.05
    - distribution="normal"
    - n_neighbours=3
    - rand_rest_p=0.03

</details>

<details>
<summary><b> StochasticHillClimbingOptimizer</b></summary>

    - epsilon=0.05
    - distribution="normal"
    - n_neighbours=3
    - rand_rest_p=0.03
    - p_accept=0.1
    - norm_factor="adaptive"

</details>

<details>
<summary><b> TabuOptimizer</b></summary>

    - epsilon=0.05
    - distribution="normal"
    - n_neighbours=3
    - rand_rest_p=0.03
    - tabu_factor=3

</details>

<details>
<summary><b> SimulatedAnnealingOptimizer</b></summary>

    - epsilon=0.05
    - distribution="normal"
    - n_neighbours=3
    - rand_rest_p=0.03
    - p_accept=0.1
    - norm_factor="adaptive"
    - annealing_rate=0.975
    - start_temp=1

</details>

<details>
<summary><b> RandomSearchOptimizer</b></summary>

</details>

<details>
<summary><b> RandomRestartHillClimbingOptimizer</b></summary>

    - epsilon=0.05
    - distribution="normal"
    - n_neighbours=3
    - rand_rest_p=0.03
    - n_iter_restart=10

</details>

<details>
<summary><b> RandomAnnealingOptimizer</b></summary>

    - epsilon=0.05
    - distribution="normal"
    - n_neighbours=3
    - rand_rest_p=0.03
    - annealing_rate=0.975
    - start_temp=1

</details>

<details>
<summary><b> ParallelTemperingOptimizer</b></summary>

    - n_iter_swap=10
    - rand_rest_p=0.03

</details>

<details>
<summary><b> ParticleSwarmOptimizer</b></summary>

    - inertia=0.5
    - cognitive_weight=0.5
    - social_weight=0.5
    - temp_weight=0.2
    - rand_rest_p=0.03

</details>

<details>
<summary><b> EvolutionStrategyOptimizer</b></summary>

    - mutation_rate=0.7
    - crossover_rate=0.3
    - rand_rest_p=0.03

</details>

<details>
<summary><b> BayesianOptimizer</b></summary>

    - gpr=gaussian_process["gp_nonlinear"]
    - xi=0.03
    - warm_start_smbo=None
    - rand_rest_p=0.03

</details>

<details>
<summary><b> TreeStructuredParzenEstimators</b></summary>

    - gamma_tpe=0.5
    - warm_start_smbo=None
    - rand_rest_p=0.03

</details>

<details>
<summary><b> DecisionTreeOptimizer</b></summary>

    - tree_regressor="extra_tree"
    - xi=0.01
    - warm_start_smbo=None
    - rand_rest_p=0.03

</details>

<details>
<summary><b> EnsembleOptimizer</b></summary>

    - estimators=[
            GradientBoostingRegressor(n_estimators=5),
            GaussianProcessRegressor(),
        ]
    - xi=0.01
    - warm_start_smbo=None
    - rand_rest_p=0.03

</details>







<br>

## Roadmap

<details>
<summary><b>v2.0.0</b>:heavy_check_mark:</summary>

  - [x] Change API
</details>

<details>
<summary><b>v2.1.0</b>:heavy_check_mark:</summary>

  - [x] Save memory of evaluations for later runs (long term memory)
  - [x] Warm start sequence based optimizers with long term memory
  - [x] Gaussian process regressors from various packages (gpy, sklearn, GPflow, ...) via wrapper
</details>

<details>
<summary><b>v2.2.0</b>:heavy_check_mark:</summary>

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
<summary><b>v2.3.0</b>:heavy_check_mark:</summary>

  - [x] Tree-structured Parzen Estimator
  - [x] Decision Tree Optimizer
  - [x] add "max_sample_size" and "skip_retrain" parameter for sbom to decrease optimization time
</details>

<details open>
<summary><b>v3.0.0</b></summary>

  - [ ] New API
      - [ ] expand usage of objective-function
      - [ ] No passing of training data into Hyperactive
      - [ ] Removing "long term memory"-support (better to do in separate package)
      - [ ] More intuitive selection of optimization strategies and parameters
      - [ ] Separate optimization algorithms into other package

</details>

<details>
<summary><b>v3.1.0</b></summary>

  - [ ] Downhill-Simplex-Method
  - [ ] add warm start for population based optimizers
</details>

<details>
<summary><b>v3.2.0</b></summary>

  - [ ] improve distributed computing abilities

</details>

<br>



## Experimental algorithms

The following algorithms are of my own design and, to my knowledge, do not yet exist in the technical literature.
If any of these algorithms already exist I would like you to share it with me in an issue.

#### Random Annealing

A combination between simulated annealing and random search.


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
