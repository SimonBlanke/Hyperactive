## Status

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
        <a href="https://app.codacy.com/project/SimonBlanke/Hyperactive/dashboard">
        <img src="https://img.shields.io/codacy/grade/acb6989093c44fb08cc3be1dd2df1be7?style=for-the-badge&logo=codacy" alt="img not loaded: try F5 :)">
        </a>
        <a href="https://codeclimate.com/github/SimonBlanke/Hyperactive">
        <img src="https://img.shields.io/codeclimate/maintainability/SimonBlanke/Hyperactive?style=for-the-badge&logo=code-climate" alt="img not loaded: try F5 :)">
        </a>
        <a href="https://scrutinizer-ci.com/g/SimonBlanke/Hyperactive/">
        <img src="https://img.shields.io/scrutinizer/quality/g/SimonBlanke/Hyperactive?style=for-the-badge&logo=scrutinizer-ci" alt="img not loaded: try F5 :)">
        </a>
        <a href="https://www.codefactor.io/repository/github/simonblanke/hyperactive">
        <img src="https://img.shields.io/codefactor/grade/github/SimonBlanke/Hyperactive?label=code%20factor&style=for-the-badge" alt="img not loaded: try F5 :)">
        </a>
      </td>
    </tr>
  </tbody>
</table>

<br>

## Main features

- Very simple but versatile API
- Thoroughly tested code base
- Compatible with <b>any python machine-learning framework</b>
- Optimize:
  - Anything from [simple models](https://simonblanke.github.io/Hyperactive/#/./examples/sklearn_examples?id=sklearn) <br/> to complex [machine-learning-pipelines](https://simonblanke.github.io/Hyperactive/#/./examples/sklearn_pipeline_example?id=sklearn-pipeline)
  - Multi-level [ensembles](https://simonblanke.github.io/Hyperactive/#/./examples/stacking_example?id=stacking)
  - [Deep neural network](https://simonblanke.github.io/Hyperactive/#/./examples/cnn_structure?id=keras-cnn-structure) architecture
  - Other [optimization techniques](./docs/examples.md) (meta-optimization)
  - Or [any function](./docs/examples.md) you can specify with this API
- Utilize state of the art [optimization techniques](https://simonblanke.github.io/Hyperactive/#/./optimizers/README?id=optimization-techniques) like:
    - Simulated annealing
    - Evolution strategy
    - Bayesian optimization
- [High performance](https://simonblanke.github.io/Hyperactive/#/./performance/README?id=performance): Optimizer time is neglectable for most models
- Choose from a variety of different [optimization extensions](https://simonblanke.github.io/Hyperactive/#/./extentions/README?id=optimization-extensions) to improve the optimization

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
        <strong>Optimization Extentions</strong>
        <img src="./_media/blue.jpg"/>
      </td>
    </tr>
    <tr/>
    <tr valign="top">
      <td>
        <a><b>Local Search:</b></a>
          <ul>
            <li><a href="https://simonblanke.github.io/Hyperactive/#/./optimizers/HillClimbing?id=Hill-Climbing">Hill Climbing</a></li>
            <li><a href="https://simonblanke.github.io/Hyperactive/#/./optimizers/StochasticHillClimbing?id=stochastic-hill-climbing">Stochastic Hill Climbing</a></li>
            <li><a href="https://simonblanke.github.io/Hyperactive/#/./optimizers/TabuSearch?id=tabu-search">Tabu Search</a></li>
         </ul><br>
        <a><b>Random Methods:</b></a>
          <ul>
            <li><a href="https://simonblanke.github.io/Hyperactive/#/./optimizers/RandomSearch?id=random-search">Random Search</a></li>
            <li><a href="https://simonblanke.github.io/Hyperactive/#/./optimizers/RandomRestartHillClimbing?id=random-restart-hill-climbing">Random Restart Hill Climbing</a></li>
            <li><a href="https://simonblanke.github.io/Hyperactive/#/./optimizers/RandomAnnealing?id=random-annealing">Random Annealing</a> [<a href="https://github.com/SimonBlanke/Hyperactive#random-annealing">*</a>] </li>
         </ul><br>
        <a><b>Markov Chain Monte Carlo:</b></a>
          <ul>
            <li><a href="https://simonblanke.github.io/Hyperactive/#/./optimizers/SimulatedAnnealing?id=simulated-annealing">Simulated Annealing</a></li>
            <li><a href="https://simonblanke.github.io/Hyperactive/#/./optimizers/StochasticTunneling?id=stochastic-tunneling">Stochastic Tunneling</li>
            <li><a href="https://simonblanke.github.io/Hyperactive/#/./optimizers/ParallelTempering?id=parallel-tempering">Parallel Tempering</a></li>
          </ul><br>
        <a><b>Population Methods:</b></a>
          <ul>
            <li><a href="https://simonblanke.github.io/Hyperactive/#/./optimizers/ParticleSwarm?id=particle-swarm-optimization">Particle Swarm Optimizer</li>
            <li><a href="https://simonblanke.github.io/Hyperactive/#/./optimizers/EvolutionStrategy?id=evolution-strategy">Evolution Strategy</a></li>
          </ul><br>
        <a><b>Sequential Methods:</b></a>
          <ul>
            <li><a href="https://simonblanke.github.io/Hyperactive/#/./optimizers/Bayesian?id=bayesian-optimization">Bayesian Optimization</a></li>
          </ul>
      </td>
      <td>
        <a><b>Machine Learning:</b></a>
          <ul>
              <li><a href="https://simonblanke.github.io/Hyperactive/#/./examples/sklearn_examples?id=sklearn">Scikit-learn</a></li>
              <li><a href="https://simonblanke.github.io/Hyperactive/#/./examples/xgboost_example?id=xgboost">XGBoost</a></li>
              <li><a href="https://simonblanke.github.io/Hyperactive/#/./examples/lightgbm_example?id=lightgbm">LightGBM</a></li>
              <li><a href="https://simonblanke.github.io/Hyperactive/#/./examples/catboost_example?id=catboost">CatBoost</a></li>
          </ul><br>
        <a><b>Deep Learning:</b></a>
          <ul>
              <li><a href="https://simonblanke.github.io/Hyperactive/#/./examples/cnn_mnist?id=keras-cnn">Keras</a></li>
          </ul><br>
        <a><b>Distribution:</b></a>
          <ul>
              <li><a href="https://simonblanke.github.io/Hyperactive/#/./examples/multiprocessing_example?id=multiprocessing">Multiprocessing</a></li>
          </ul>
      </td>
      <td>
        <a><b>Position Initialization:</b></a>
          <ul>
            <li><a href="https://simonblanke.github.io/Hyperactive/#/./extentions/README?id=scatter-initialization">Scatter-Initialization</a> [<a href="https://github.com/SimonBlanke/Hyperactive#scatter-initialization">*</a>] </li>
            <li><a href="https://simonblanke.github.io/Hyperactive/#/./extentions/README?id=warm-start">Warm-start</a></li>
            <li>Meta-Learn (coming soon)</li>
          </ul><br>
        <a><b>Resource Allocation:</b></a>
          <ul>
            <li><a href="https://simonblanke.github.io/Hyperactive/#/./extentions/README?id=memory">Memory</a></li>
            <li>Proxy Datasets [<a href="https://github.com/SimonBlanke/Hyperactive#1-proxy-datasets-for-training-convolutional-neural-networks">1</a>]
 (coming soon)</li>
          </ul>
      </td>
    </tr>
  </tbody>
</table>

!> **Disclaimer:** The classification into the categories above is not necessarly scientifically accurate, but aims to provide an idea of the functionality of the methods.

### Experimental Algorithms

?> **Disclaimer:** The following algorithms are of my own design and, to my knowledge, do not yet exist in the technical literature.
If any of these algorithms already exist I would like to ask you to share it with me in an issue.

**Random Annealing**

A combination between simulated annealing and random search.

**Scatter Initialization**

Inspired by hyperband optimization.

<br>

### References

#### [1] [Proxy Datasets for Training Convolutional Neural Networks](https://arxiv.org/pdf/1906.04887v1.pdf)

<br>

### License

[![LICENSE](https://img.shields.io/github/license/SimonBlanke/Hyperactive?style=for-the-badge)](https://github.com/SimonBlanke/Hyperactive/blob/master/LICENSE)
