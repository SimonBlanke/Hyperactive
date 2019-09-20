<p align="center">
  <br>
  <a href="https://simonblanke.github.io/Hyperactive/"><img src="./images/hyperactive_logo.png" height="200"></a>
  <br>
</p>

<br>

---

<h2 align="center">A hyperparameter optimization and meta-learning toolbox for convenient and fast prototyping of machine-/deep-learning models.</h2>

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
        <img src="https://img.shields.io/codefactor/grade/github/SimonBlanke/Hyperactive?label=code%20factor&style=flat-square" alt="img not loaded: try F5 :)">
        </a>     
      </td>
    </tr>
  </tbody>
</table>

<br>

---

<div align="center"><a name="menu"></a>
  <h3>
    <a href="https://simonblanke.github.io/Hyperactive/">Documentation</a> |
    <a href="https://github.com/SimonBlanke/Hyperactive#overview">Overview</a> |
    <a href="https://github.com/SimonBlanke/Hyperactive#installation">Installation</a> |
    <a href="https://github.com/SimonBlanke/Hyperactive#license">License</a>
  </h3>
</div>

---

<br>

## Overview

- Very simple but powerful API
- Compatible with <b>any python machine-learning framework</b>
- Optimize:
  - Complex machine learning pipelines 
  - Multi-level ensembles
  - Deep neural network architecture
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
        <img src="images/blue.jpg"/>
      </td>
      <td>
        <strong>Tested and Supported Packages</strong>
        <img src="images/blue.jpg"/>
      </td>
      <td>
        <strong>Optimization Extentions</strong>
        <img src="images/blue.jpg"/>
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

Hyperactive (stable) is available on PyPi:
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

