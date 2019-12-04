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
        <a href="https://app.codacy.com/project/SimonBlanke/Hyperactive/dashboard">
        <img src="https://img.shields.io/codacy/grade/acb6989093c44fb08cc3be1dd2df1be7?style=flat-square&logo=codacy" alt="img not loaded: try F5 :)">
        </a>
        <a href="https://codeclimate.com/github/SimonBlanke/Hyperactive">
        <img src="https://img.shields.io/codeclimate/maintainability/SimonBlanke/Hyperactive?style=flat-square&logo=code-climate" alt="img not loaded: try F5 :)">
        </a>
        <a href="https://scrutinizer-ci.com/g/SimonBlanke/Hyperactive/">
        <img src="https://img.shields.io/scrutinizer/quality/g/SimonBlanke/Hyperactive?style=flat-square&logo=scrutinizer-ci" alt="img not loaded: try F5 :)">
        </a>
        <a href="https://www.codefactor.io/repository/github/simonblanke/hyperactive">
        <img src="https://img.shields.io/codefactor/grade/github/SimonBlanke/Hyperactive?label=code%20factor&style=flat-square&logo=codefactor" alt="img not loaded: try F5 :)">
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

---

<div align="center"><a name="menu"></a>
  <h3>
    <a href="https://simonblanke.github.io/Hyperactive/">Documentation</a> •
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
    - [Deep neural network](https://simonblanke.github.io/Hyperactive/#/./examples/use_cases?id=keras-cnn-structure) architecture
    - Other [optimization techniques](https://simonblanke.github.io/Hyperactive/#/./examples/use_cases?id=meta-optimization) (meta-optimization)
    - Or [any function](https://simonblanke.github.io/Hyperactive/#/./examples/math_functions?id=rosenbrock-function) you can specify with this API
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
        <strong>Optimization Extentions</strong>
        <img src="./docs/_media/blue.jpg"/>
      </td>
    </tr>
    <tr/>
    <tr valign="top">
      <td>
        <a><b>Local Search:</b></a>
          <ul>
            <li><a href="https://simonblanke.github.io/Hyperactive/#/./optimizers/local_search?id=hill-climbing">Hill Climbing</a></li>
            <li><a href="https://simonblanke.github.io/Hyperactive/#/./optimizers/local_search?id=stochastic-hill-climbing">Stochastic Hill Climbing</a></li>
            <li><a href="https://simonblanke.github.io/Hyperactive/#/./optimizers/local_search?id=tabu-search">Tabu Search</a></li>
         </ul>
        <a><b>Random Methods:</b></a>
          <ul>
            <li><a href="https://simonblanke.github.io/Hyperactive/#/./optimizers/random_methods?id=random-search">Random Search</a></li>
            <li><a href="https://simonblanke.github.io/Hyperactive/#/./optimizers/random_methods?id=random-restart-hill-climbing">Random Restart Hill Climbing</a></li>
            <li><a href="https://simonblanke.github.io/Hyperactive/#/./optimizers/random_methods?id=random-annealing">Random Annealing</a> [<a href="https://github.com/SimonBlanke/Hyperactive#random-annealing">*</a>] </li>
         </ul>
        <a><b>Markov Chain Monte Carlo:</b></a>
          <ul>
            <li><a href="https://simonblanke.github.io/Hyperactive/#/./optimizers/mcmc?id=simulated-annealing">Simulated Annealing</a></li>
            <li><a href="https://simonblanke.github.io/Hyperactive/#/./optimizers/mcmc?id=stochastic-tunneling">Stochastic Tunneling</li>
            <li><a href="https://simonblanke.github.io/Hyperactive/#/./optimizers/mcmc?id=parallel-tempering">Parallel Tempering</a></li>
          </ul>
        <a><b>Population Methods:</b></a>
          <ul>
            <li><a href="https://simonblanke.github.io/Hyperactive/#/./optimizers/population_methods?id=particle-swarm-optimization">Particle Swarm Optimizer</li>
            <li><a href="https://simonblanke.github.io/Hyperactive/#/./optimizers/population_methods?id=evolution-strategy">Evolution Strategy</a></li>
          </ul>
        <a><b>Sequential Methods:</b></a>
          <ul>
            <li><a href="https://simonblanke.github.io/Hyperactive/#/./optimizers/sequential_methods?id=bayesian-optimization">Bayesian Optimization</a></li>
          </ul>
      </td>
      <td>
        <a><b>Machine Learning:</b></a>
          <ul>
              <li><a href="https://simonblanke.github.io/Hyperactive/#/./examples/machine_learning?id=sklearn">Scikit-learn</a></li>
              <li><a href="https://simonblanke.github.io/Hyperactive/#/./examples/machine_learning?id=xgboost">XGBoost</a></li>
              <li><a href="https://simonblanke.github.io/Hyperactive/#/./examples/machine_learning?id=lightgbm">LightGBM</a></li>
              <li><a href="https://simonblanke.github.io/Hyperactive/#/./examples/machine_learning?id=catboost">CatBoost</a></li>
              <li><a href="https://simonblanke.github.io/Hyperactive/#/./examples/machine_learning?id=rgf">RGF</a></li>
              <li><a href="https://simonblanke.github.io/Hyperactive/#/./examples/machine_learning?id=mlxtend">Mlxtend</a></li>
          </ul>
        <a><b>Deep Learning:</b></a>
          <ul>
              <li><a href="https://simonblanke.github.io/Hyperactive/#/./examples/deep_learning?id=tensorflow">Tensorflow</a></li>
              <li><a href="https://simonblanke.github.io/Hyperactive/#/./examples/deep_learning?id=keras-cnn">Keras</a></li>
          </ul>
        <a><b>Distribution:</b></a>
          <ul>
              <li><a href="https://simonblanke.github.io/Hyperactive/#/./examples/distribution?id=multiprocessing">Multiprocessing</a></li>
              <li>Ray</li>
          </ul>
      </td>
      <td>
        <a><b>Position Initialization:</b></a>
          <ul>
            <li><a href="https://simonblanke.github.io/Hyperactive/#/./extentions/extensions?id=scatter-initialization">Scatter-Initialization</a> [<a href="https://github.com/SimonBlanke/Hyperactive#scatter-initialization">*</a>] </li>
            <li><a href="https://simonblanke.github.io/Hyperactive/#/./extentions/extensions?id=warm-start">Warm-start</a></li>
          </ul>
        <a><b>Resource Allocation:</b></a>
          <ul>
            <li><a href="https://simonblanke.github.io/Hyperactive/#/./extentions/extensions?id=memory">Memory</a></li>
               <ul>
                 <li>Short term</li>
                 <li>Long term</li>
               </ul>
          </ul>
      </td>
    </tr>
  </tbody>
</table>

<div align="center">
  <h3>
    This readme provides only a short introduction. For more information check out the <br/>
    <a href="https://simonblanke.github.io/Hyperactive/">full documentation</a>
  </h3>
</div>

<br>

## Installation
[![PyPI version](https://badge.fury.io/py/hyperactive.svg)](https://badge.fury.io/py/hyperactive)

The most recent version of Hyperactive is available on PyPi:
```console
pip install hyperactive
```

<br>

## Roadmap

<details open>
<summary><b>v2.0.0</b></summary> 
  
  - [x] Change API
  - [x] Ray integration 
</details>

<details>
<summary><b>v2.1.0</b></summary>
  
  - [ ] Save memory of evals for later runs 
  - [ ] Warm start sequence based optimizers with memory from previous runs
</details>

<details>
<summary><b>v2.2.0</b></summary>
  
  - [ ] Tree-structured Parzen Estimator
  - [ ] Spiral optimization
  - [ ] Downhill-Simplex-Method
</details>

<br>

## Experimental algorithms

The following algorithms are of my own design and, to my knowledge, do not yet exist in the technical literature.
If any of these algorithms still exist I ask you to share it with me in an issue.

#### Random Annealing

A combination between simulated annealing and random search.

#### Scatter Initialization

Inspired by hyperband optimization.

<br>

## References

#### [1] [Proxy Datasets for Training Convolutional Neural Networks](https://arxiv.org/pdf/1906.04887v1.pdf)

<br>

## License

[![LICENSE](https://img.shields.io/github/license/SimonBlanke/Hyperactive?style=for-the-badge)](https://github.com/SimonBlanke/Hyperactive/blob/master/LICENSE)
