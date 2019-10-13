## Choose an optimizer

Your decision to use a specific optimizer should be based on the time it takes to evaluate a model and if you already have a start point. Try to stick to the following <b>guidelines</b>, when choosing an optimizer:
- only use local or mcmc optimizers, if you have a <b>good start point</b>
- random optimizers are a good way to <b>start exploring</b> the search space
- the majority of the <b>iteration-time</b> should be the <b>evaluation-time</b> of the model

All optimization techniques are explained in more detail [here](https://simonblanke.github.io/Hyperactive/#/./optimizers/README?id=optimization-techniques). A comparison between the iteration- and evaluation-time for different models can be seen [here](https://simonblanke.github.io/Hyperactive/#/./performance/README?id=performance).
You can choose the optimizer by passing one of the following strings to the 'optimizer' keyword in the Hyperactive-class:

  - "HillClimbing"
  - "StochasticHillClimbing"
  - "TabuSearch"
  - "RandomSearch"
  - "RandomRestartHillClimbing"
  - "RandomAnnealing"
  - "SimulatedAnnealing",
  - "StochasticTunneling"
  - "ParallelTempering"
  - "ParticleSwarm"
  - "EvolutionStrategy"
  - "Bayesian"
