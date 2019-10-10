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
            <li><a href="https://simonblanke.github.io/Hyperactive/#/./optimizers/README?id=hill-climbing">Hill Climbing</a></li>
            <li><a href="https://simonblanke.github.io/Hyperactive/#/./optimizers/README?id=stochastic-hill-climbing">Stochastic Hill Climbing</a></li>
            <li><a href="https://simonblanke.github.io/Hyperactive/#/./optimizers/README?id=tabu-search">Tabu Search</a></li>
         </ul><br>
        <a><b>Random Methods:</b></a>
          <ul>
            <li><a href="https://simonblanke.github.io/Hyperactive/#/./optimizers/README?id=random-search">Random Search</a></li>
            <li><a href="https://simonblanke.github.io/Hyperactive/#/./optimizers/README?id=random-restart-hill-climbing">Random Restart Hill Climbing</a></li>
            <li><a href="https://simonblanke.github.io/Hyperactive/#/./optimizers/README?id=random-annealing">Random Annealing</a> [<a href="https://github.com/SimonBlanke/Hyperactive#random-annealing">*</a>] </li>
         </ul><br>
        <a><b>Markov Chain Monte Carlo:</b></a>
          <ul>
            <li><a href="https://simonblanke.github.io/Hyperactive/#/./optimizers/README?id=simulated-annealing">Simulated Annealing</a></li>
            <li><a href="https://simonblanke.github.io/Hyperactive/#/./optimizers/README?id=stochastic-tunneling">Stochastic Tunneling</li>
            <li><a href="https://simonblanke.github.io/Hyperactive/#/./optimizers/README?id=parallel-tempering">Parallel Tempering</a></li>
          </ul><br>
        <a><b>Population Methods:</b></a>
          <ul>
            <li><a href="https://simonblanke.github.io/Hyperactive/#/./optimizers/README?id=particle-swarm-optimization">Particle Swarm Optimizer</li>
            <li><a href="https://simonblanke.github.io/Hyperactive/#/./optimizers/README?id=evolution-strategy">Evolution Strategy</a></li>
          </ul><br>
        <a><b>Sequential Methods:</b></a>
          <ul>
            <li><a href="https://simonblanke.github.io/Hyperactive/#/./optimizers/README?id=bayesian-optimization">Bayesian Optimization</a></li>
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
