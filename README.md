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

Hyperactive is primarly a <b>hyperparameter optimization toolkit</b>, that aims to simplify the model-selection and -tuning process. You can use any machine- or deep-learning package and it is not necessary to learn new syntax. Hyperactive offers <b>high versatility</b> in model optimization because of two characteristics:

  - You can define any kind of model in the objective function. It just has to return a score/metric that gets maximized.
  - The search space accepts not just 'int', 'float' or 'str' as data types but even functions, classes or any python objects.

A large part of the Hyperactive backend is developed and tested in separate repositories. If you want to take a look at the sourcecode, you can find them in the following repositories:
  - Optimizer modules: <a href="https://github.com/SimonBlanke/Gradient-Free-Optimizers">Gradient-Free-Optimizers</a>
  - Meta-data storage: <a href="https://github.com/SimonBlanke/Optimization-Metadata">Optimization-Metadata</a>


<br>

<div align="center">
  <h3>
    For more information, visualization and details about the API check out the <br/>
    <a href="https://simonblanke.github.io/Hyperactive/">website</a>
  </h3>
</div>

---

<div align="center"><a name="menu"></a>
  <h3>
    <a href="https://github.com/SimonBlanke/Hyperactive#main-features">Main features</a> •
    <a href="https://github.com/SimonBlanke/Hyperactive#installation">Installation</a> •
    <a href="https://github.com/SimonBlanke/Hyperactive#roadmap">Roadmap</a> •
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
        <strong>Optimization Extensions</strong>
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
              <li><a href="https://simonblanke.github.io/Hyperactive#/./examples/distribution?id=ray">Ray</a></li>
          </ul>
      </td>
      <td>
        <a><b>Position Initialization:</b></a>
          <ul>
            <li><a href="https://simonblanke.github.io/Hyperactive#/./examples/extensions?id=scatter-initialization">Scatter-Initialization</a> [<a href="#/./overview#experimental-algorithms">*</a>] </li>
            <li><a href="https://simonblanke.github.io/Hyperactive#/./examples/extensions?id=warm-start">Warm-start</a></li>
          </ul>
        <a><b>Resource Allocation:</b></a>
          <ul>
            <li><a href="https://simonblanke.github.io/Hyperactive#/./examples/extensions?id=memory">Memory</a></li>
               <ul>
                 <li>Short term</li>
                 <li>Long term</li>
               </ul>
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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from hyperactive import Hyperactive

data = load_breast_cancer()
X, y = data.data, data.target

'''define the model in a function'''
def model(para, X, y):
    '''optimize one or multiple hyperparameters'''
    gbc = GradientBoostingClassifier(n_estimators=para['n_estimators'])
    scores = cross_val_score(gbc, X, y)

    return scores.mean()

'''create the search space and search_config'''
search_config = {
    model: {'n_estimators': range(10, 200, 10)}
}

'''start the optimization run'''
opt = Hyperactive(X, y)
opt.search(search_config, n_iter=20)
```


<br>

## Roadmap

<details>
<summary><b>v2.0.0</b>:heavy_check_mark:</summary>

  - [x] Change API
  - [x] Ray integration
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
      - [ ] improve distributed computing abilities
      - [ ] separate optimizer and n_iter for each job
      - [ ] expand usage of objective-function

</details>

<details>
<summary><b>v3.1.0</b></summary>

  - [ ] Spiral optimization
  - [ ] Downhill-Simplex-Method
  - [ ] upgrade particle swarm optimization
  - [ ] upgrade evolution strategy
  - [ ] add warm start for population based optimizers
  - [ ] Meta-Optimization of local optimizers
</details>

<br>

## Experimental algorithms

The following algorithms are of my own design and, to my knowledge, do not yet exist in the technical literature.
If any of these algorithms already exist I would like you to share it with me in an issue.

#### Random Annealing

A combination between simulated annealing and random search.

#### Scatter Initialization

Inspired by hyperband optimization.

<br>

## References

#### [dto] [Scikit-Optimize](https://github.com/scikit-optimize/scikit-optimize/blob/master/skopt/learning/forest.py)

<br>

## License

[![LICENSE](https://img.shields.io/github/license/SimonBlanke/Hyperactive?style=for-the-badge)](https://github.com/SimonBlanke/Hyperactive/blob/master/LICENSE)
