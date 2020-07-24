## Project Status

<table>
  <tbody>
    <tr align="left" valign="center">
      <td>
        <strong>Master status:</strong>
      </td>
      <td>
        <a href="https://travis-ci.com/SimonBlanke/Hyperactive">
          <img src="https://img.shields.io/travis/com/SimonBlanke/Hyperactive/master?style=for-the-badge&logo=travis" alt="img not loaded: try F5 :)">
        </a>
        <a href="https://coveralls.io/github/SimonBlanke/Hyperactive">
          <img src="https://img.shields.io/coveralls/github/SimonBlanke/Hyperactive?style=for-the-badge&logo=codecov" alt="img not loaded: try F5 :)">
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
          <img src="https://img.shields.io/travis/SimonBlanke/Hyperactive/dev?style=for-the-badge&logo=travis" alt="img not loaded: try F5 :)">
        </a>
        <a href="https://coveralls.io/github/SimonBlanke/Hyperactive?branch=dev">
          <img src="https://img.shields.io/coveralls/github/SimonBlanke/Hyperactive/dev?style=for-the-badge&logo=codecov" alt="img not loaded: try F5 :)">
        </a>
      </td>
    </tr>
    <tr/>    <tr align="left" valign="center">
      <td>
         <strong>Code quality:</strong>
      </td>
      <td>
        <a href="https://codeclimate.com/github/SimonBlanke/Hyperactive">
        <img src="https://img.shields.io/codeclimate/maintainability/SimonBlanke/Hyperactive?style=for-the-badge&logo=code-climate" alt="img not loaded: try F5 :)">
        </a>
        <a href="https://scrutinizer-ci.com/g/SimonBlanke/Hyperactive/">
        <img src="https://img.shields.io/scrutinizer/quality/g/SimonBlanke/Hyperactive?style=for-the-badge&logo=scrutinizer-ci" alt="img not loaded: try F5 :)">
        </a>
      </td>
    </tr>
    <tr/>    <tr align="left" valign="center">
      <td>
        <strong>Latest versions:</strong>
      </td>
      <td>
        <a href="https://github.com/SimonBlanke/Hyperactive/releases">
          <img src="https://img.shields.io/github/v/release/SimonBlanke/Hyperactive?style=for-the-badge&logo=github" alt="img not loaded: try F5 :)">
        </a>
        <a href="https://pypi.org/project/hyperactive/">
          <img src="https://img.shields.io/pypi/v/Hyperactive?style=for-the-badge&logo=PyPi&logoColor=white" alt="img not loaded: try F5 :)">
        </a>
      </td>
    </tr>
  </tbody>
</table>

<br>

## Main Features

- Thoroughly tested code base
- Compatible with <b>any python machine-learning framework</b>
- Optimize:
  - Anything from [simple models](#/./examples/machine_learning?id=sklearn) <br/> to complex [machine-learning-pipelines](#/./examples/use_cases?id=sklearn-pipeline)
  - Multi-level [ensembles](#/./examples/use_cases?id=stacking)
  - [Deep neural network](#/./examples/use_cases?id=neural-architecture-search) architecture
  - Other [optimization techniques](#/./examples/use_cases?id=meta-optimization) (meta-optimization)
  - Or [any function](#/./examples/test_functions?id=rosenbrock-function) you can specify with this API
- Utilize state of the art optimization techniques like:
  - Simulated annealing
  - Evolution strategy
  - Bayesian optimization
- [High performance](#/./performance?id=performance): Optimizer time is neglectable for most models
- Choose from a variety of different [optimization extensions](#/./examples/extensions) to improve the optimization

<br>

<br>

<table>
  <tbody>
    <tr align="center" valign="center">
      <td>
        <strong>Optimization Techniques</strong>
        <img src="./_media/blue.jpg"/>
      </td>
      <td>
        <strong>Tested and Supported Packages</strong>
        <img src="./_media/blue.jpg"/>
      </td>
      <td>
        <strong>Optimization Extensions</strong>
        <img src="./_media/blue.jpg"/>
      </td>
    </tr>
    <tr/>
    <tr valign="top">
      <td>
        <a><b>Local Search:</b></a>
          <ul>
            <li><a href="#/./optimizers/local_search?id=Hill-Climbing">Hill Climbing</a></li>
            <li><a href="#/./optimizers/local_search?id=stochastic-hill-climbing">Stochastic Hill Climbing</a></li>
            <li><a href="#/./optimizers/local_search?id=tabu-search">Tabu Search</a></li>
            <li><a href="#/./optimizers/mcmc?id=simulated-annealing">Simulated Annealing</a></li>
         </ul><br>
        <a><b>Global Search:</b></a>
          <ul>
            <li><a href="#/./optimizers/random_methods?id=random-search">Random Search</a></li>
            <li><a href="#/./optimizers/random_methods?id=random-restart-hill-climbing">Random Restart Hill Climbing</a></li>
            <li><a href="#/./optimizers/random_methods?id=random-annealing">Random Annealing</a> [<a href="#/./overview#experimental-algorithms">*</a>] </li>
         </ul><br>
        <a><b>Population Methods:</b></a>
          <ul>
            <li><a href="#/./optimizers/mcmc?id=parallel-tempering">Parallel Tempering</a></li>
            <li><a href="#/./optimizers/population_methods?id=particle-swarm-optimization">Particle Swarm Optimizer</li>
            <li><a href="#/./optimizers/population_methods?id=evolution-strategy">Evolution Strategy</a></li>
          </ul><br>
        <a><b>Sequential Methods:</b></a>
          <ul>
            <li><a href="#/./optimizers/sequential_methods?id=bayesian-optimization">Bayesian Optimization</a></li>
            <li><a href="https://simonblanke.github.io/Hyperactive/#/./optimizers/sequential_methods?id=tree-of-parzen-estimators">Tree of Parzen Estimators</a></li>
            <li><a href="https://simonblanke.github.io/Hyperactive/#/./optimizers/sequential_methods?id=decision-tree-optimizer">Decision Tree Optimizer</a>
            [<a href="#/./overview#references">dto</a>] </li>
          </ul>
      </td>
      <td>
        <a><b>Machine Learning:</b></a>
          <ul>
              <li><a href="#/./examples/machine_learning?id=sklearn">Scikit-learn</a></li>
              <li><a href="#/./examples/machine_learning?id=xgboost">XGBoost</a></li>
              <li><a href="#/./examples/machine_learning?id=lightgbm">LightGBM</a></li>
              <li><a href="#/./examples/machine_learning?id=catboost">CatBoost</a></li>
              <li><a href="#/./examples/machine_learning?id=rgf">RGF</a></li>
              <li><a href="#/./examples/machine_learning?id=mlxtend">Mlxtend</a></li>
          </ul><br>
        <a><b>Deep Learning:</b></a>
          <ul>
              <li><a href="#/./examples/deep_learning?id=tensorflow">Tensorflow</a></li>
              <li><a href="#/./examples/deep_learning?id=keras-cnn">Keras</a></li>
              <li>Pytorch</li>
          </ul><br>
        <a><b>Distribution:</b></a>
          <ul>
              <li><a href="#/./examples/distribution?id=multiprocessing">Multiprocessing</a></li>
          </ul>
      </td>
      <td>
        <a><b>Position Initialization:</b></a>
          <ul>
            <li><a href="#/./examples/extensions?id=warm-start">Warm-start</a></li>
          </ul>
        <a><b>Resource Allocation:</b></a>
          <ul>
            <li><a href="#/./examples/extensions?id=memory">Memory</a></li>
               <ul>
                 <li>Short term</li>
                 <li>Long term</li>
               </ul>
          </ul>
      </td>
    </tr>
  </tbody>
</table>

!> **Disclaimer:** The classification into the categories above is not necessarly scientifically accurate, but aims to provide an idea of the functionality of the methods.

## Installation Options

**Hyperactive (stable) is available on PyPi:**

```bash
pip install hyperactive
```

<br>

**Hyperactive (master) from Github:**

```bash
git clone https://github.com/SimonBlanke/Hyperactive.git
pip install Hyperactive/
```

<br>

**Hyperactive (dev version) from Github:**

```bash
git clone https://github.com/SimonBlanke/Hyperactive/tree/dev.git
pip install Hyperactive/
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
      - [ ] separate optimizer and n_iter for each job
      - [ ] expand usage of objective-function
  - [x] Simpler and faster meta-data collection, saving and loading

</details>

<details>
<summary><b>v3.1.0</b></summary>

  - [ ] Spiral optimization
  - [ ] Downhill-Simplex-Method
  - [ ] upgrade particle swarm optimization
  - [ ] upgrade evolution strategy
  - [ ] add warm start for population based optimizers
  - [ ] Meta-Optimization of local optimizers
  - [ ] improve distributed computing abilities
</details>

<br><br>

#### Experimental Algorithms

?> **Disclaimer:** The following algorithms are of my own design and, to my knowledge, do not yet exist in the technical literature.
If any of these algorithms already exist I would like you to share it with me in an issue.

**Random Annealing**

A combination between simulated annealing and random search.

<br>

#### References

[dto] [Scikit-Optimize](https://github.com/scikit-optimize/scikit-optimize/blob/master/skopt/learning/forest.py)

<br>


#### Citing Hyperactive

    @Misc{hyperactive2019,
      author =   {{Simon Blanke}},
      title =    {{Hyperactive}: A hyperparameter optimization and meta-learning toolbox for machine-/deep-learning models.},
      howpublished = {\url{https://github.com/SimonBlanke}},
      year = {since 2019}
    }


<br>


#### License

[![LICENSE](https://img.shields.io/github/license/SimonBlanke/Hyperactive?style=for-the-badge)](https://github.com/SimonBlanke/Hyperactive/blob/master/LICENSE)
