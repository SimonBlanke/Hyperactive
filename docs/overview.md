## Main features

- Very simple but versatile API
- Thoroughly tested code base
- Compatible with <b>any python machine-learning framework</b>
- Optimize:
  - Anything from [simple models](./examples.md) <br/> to complex [machine-learning-pipelines](./examples.md)
  - Multi-level [ensembles](./examples.md)
  - [Deep neural network](./examples.md) architecture
  - Other [optimization techniques](./examples.md) (meta-optimization)
  - Or [any function](./examples.md) you can specify with this API
- Utilize state of the art [optimization techniques](https://github.com/SimonBlanke/Hyperactive/tree/master/hyperactive/optimizers#optimization-techniques) like:
    - Simulated annealing
    - Evolution strategy
    - Bayesian optimization
- [High performance](https://github.com/SimonBlanke/Hyperactive/tree/master/plots#performance): Optimizer time is neglectable for most models
- Choose from a variety of different [optimization extensions](https://github.com/SimonBlanke/Hyperactive/tree/master/hyperactive#advanced-features) to improve the optimization

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
            <li><a href="https://github.com/SimonBlanke/Hyperactive/tree/master/hyperactive/optimizers#hill-climbing">Hill Climbing</a></li>
            <li><a href="https://github.com/SimonBlanke/Hyperactive/tree/master/hyperactive/optimizers#stochastic-hill-climbing">Stochastic Hill Climbing</a></li>
            <li><a href="https://github.com/SimonBlanke/Hyperactive/tree/master/hyperactive/optimizers#tabu-search">Tabu Search</a></li>
         </ul>
        <a><b>Random Methods:</b></a>
          <ul>
            <li><a href="https://github.com/SimonBlanke/Hyperactive/tree/master/hyperactive/optimizers#random-search">Random Search</a></li>
            <li><a href="https://github.com/SimonBlanke/Hyperactive/tree/master/hyperactive/optimizers#random-restart-hill-climbing">Random Restart Hill Climbing</a></li>
            <li><a href="https://github.com/SimonBlanke/Hyperactive/tree/master/hyperactive/optimizers#random-annealing">Random Annealing</a> [<a href="https://github.com/SimonBlanke/Hyperactive#random-annealing">*</a>] </li> 
         </ul>
        <a><b>Markov Chain Monte Carlo:</b></a>
          <ul>
            <li><a href="https://github.com/SimonBlanke/Hyperactive/tree/master/hyperactive/optimizers#simulated-annealing">Simulated Annealing</a></li>
            <li><a href="https://github.com/SimonBlanke/Hyperactive/tree/master/hyperactive/optimizers#stochastic-tunneling">Stochastic Tunneling</li>
            <li><a href="https://github.com/SimonBlanke/Hyperactive/tree/master/hyperactive/optimizers#parallel-tempering">Parallel Tempering</a></li>
          </ul>
        <a><b>Population Methods:</b></a>
          <ul>
            <li><a href="https://github.com/SimonBlanke/Hyperactive/tree/master/hyperactive/optimizers#particle-swarm-optimization">Particle Swarm Optimizer</li>
            <li><a href="https://github.com/SimonBlanke/Hyperactive/tree/master/hyperactive/optimizers#evolution-strategy">Evolution Strategy</a></li>
          </ul>
        <a><b>Sequential Methods:</b></a>
          <ul>
            <li><a href="https://github.com/SimonBlanke/Hyperactive/tree/master/hyperactive/optimizers#bayesian-optimization">Bayesian Optimization</a></li>
          </ul>
      </td>
      <td>
        <a><b>Machine Learning:</b></a>
          <ul>
              <li><a href="https://github.com/SimonBlanke/Hyperactive/blob/master/examples/machine_learning/sklearn_.py">Scikit-learn</a></li>
              <li><a href="https://github.com/SimonBlanke/Hyperactive/blob/master/examples/machine_learning/xgboost_.py">XGBoost</a></li>
              <li><a href="https://github.com/SimonBlanke/Hyperactive/blob/master/examples/machine_learning/lightgbm_.py">LightGBM</a></li>
              <li><a href="https://github.com/SimonBlanke/Hyperactive/blob/master/examples/machine_learning/catboost_.py">CatBoost</a></li>
          </ul>
        <a><b>Deep Learning:</b></a>
          <ul>
              <li><a href="https://github.com/SimonBlanke/Hyperactive/tree/master/examples/deep_learning">Keras</a></li>
          </ul>
        <a><b>Distribution:</b></a>
          <ul>
              <li><a href="https://github.com/SimonBlanke/Hyperactive/blob/master/examples/distribution/multiprocessing_.py">Multiprocessing</a></li>
          </ul>
      </td>
      <td>
        <a><b>Position Initialization:</b></a>
          <ul>
            <li><a href="https://github.com/SimonBlanke/Hyperactive/tree/master/hyperactive#scatter-initialization">Scatter-Initialization</a> [<a href="https://github.com/SimonBlanke/Hyperactive#scatter-initialization">*</a>] </li>
            <li><a href="https://github.com/SimonBlanke/Hyperactive/tree/master/hyperactive#warm-start">Warm-start</a></li>
            <li>Meta-Learn (coming soon)</li>
          </ul>
        <a><b>Resource Allocation:</b></a>
          <ul>
            <li><a href="https://github.com/SimonBlanke/Hyperactive/tree/master/hyperactive#memory">Memory</a></li>
            <li>Proxy Datasets [<a href="https://github.com/SimonBlanke/Hyperactive#1-proxy-datasets-for-training-convolutional-neural-networks">1</a>]
 (coming soon)</li>
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
