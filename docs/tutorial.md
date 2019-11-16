## Introduction


## Create the search space

Since v1.0.0 the search space is created by defining:
  - a <b>function</b> for the model
  - a parameter <b>dictionary</b>


The function receives 3 arguments:
  - <b>para</b> : This defines what part of the model-function should be optimized
  - <b>X</b> : Training features
  - <b>y</b> : Training target

 The function should return some kind of metric that will be <b>maximized</b> during the search.

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

def model(para, X, y):
    model = GradientBoostingClassifier(
        n_estimators=para["n_estimators"],
        max_depth=para["max_depth"],
    )
    scores = cross_val_score(model, X, y, cv=3)

    return scores.mean()
```

The search_config is a dictionary, that has the <b>model-function as a key</b> and its <b>values defines the search space</b> for this model. The search space is an additional dictionary that will be used in 'para' within the model-function.

```python
search_config = {
    model: {
        "n_estimators": range(10, 200, 10),
        "max_depth": range(2, 12),
        "min_samples_split": range(2, 12),
    }
}
```

This way of creating the search space has <b>multiple advantages</b>:
  - No new syntax to learn. You can create the model as you are used to.
  - It makes the usage of hyperactive very versatile, because you can define <b>any kind of function</b> and optimize it. This enables:
    - The optimization of:
      - complex machine-learning pipelines and ensembles
      - deep neural network architecture
    - The usage of <b>any machine learning framework</b> you like. The following are tested:
      - Sklearn
      - XGBoost
      - LightGBM
      - CatBoost
      - Keras


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


## How many iterations?

The number of iterations should be low for your first optimization to get to know the iteration-time.
For the <b>iteration-time</b> you should take the following effects into account:
- A <b>k-fold-crossvalidation</b> increases evaluation-time like training on k-1 times on the training data
- If you lower <b>cv below 1</b> the evaluation will deal with it like a training/validation-split, where cv marks the training data fraction. Therefore lower cv means faster evaluation.
- Some optimizers will do (and need) <b>multiple evaluations</b> per iteration:
  - Particle-swarm-optimization
  - Evoluion strategy
  - Parallel Tempering
- The <b>complexity</b> of the machine-/deep-learning models will heavily influence the evaluation- and therefore iteration-time.
- The <b>number of epochs</b> should probably be kept low. You just want to compare different types of models. Retrain the best model afterwards with more epochs.

## Distribution

You can start multiple optimizations in <b>parallel</b> by increasing the number of jobs. This can make sense if you want to increase the chance of finding the optimal solution or optimize different models at the same time.

## Position initialization

#### Scatter-Initialization

This technique was inspired by the 'Hyperband Optimization' and aims to find a good initial position for the optimization. It does so by evaluating n random positions with a training subset of 1/n the size of the original dataset. The position that achieves the best score is used as the starting position for the optimization.


#### Warm-Start

When a search is finished the warm-start-dictionary for the best position in the hyperparameter search space (and its metric) is printed in the command line (at verbosity=1). If multiple searches ran in parallel the warm-start-dictionaries are sorted by the best metric in decreasing order. If the start position in the warm-start-dictionary is not within the search space defined in the search_config an error will occure.

## Resources allocation

#### Memory
After the evaluation of a model the position (in the hyperparameter search dictionary) and the cross-validation score are written to a dictionary. If the optimizer tries to evaluate this position again it can quickly lookup if a score for this position is present and use it instead of going through the extensive training and prediction process.
