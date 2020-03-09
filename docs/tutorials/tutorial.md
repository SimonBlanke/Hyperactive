## Introduction

Hyperactive is primarily designed to optimize **hyperparameters of machine learning models**, but it can be used to optimize any other type of "model" that returns a fitness value. <br>

In general hyperactive works by searching through a set of parameters of an **objective function** (or model function). The objective function returns a fitness value that gets maximized during the optimization process. The **search space** defines the range of parameters that will be searched during the optimization process. <br>

> The following chapters provide a step by step explanation of how to start your first optimization run. Alternatively there are plenty of examples to learn how to use hyperactive.

<br>

## The search-config parameter

The search-config is a parameter of Hyperactive that contains the objective function(s) and search-space(s) of the optimization run. It therefore defines what model to evaluate and which hyperparameters to search through.

Since v1.0.0 the search-config is created by defining:
  - a **function** for the **model**:
  ```python
  def model(para, X, y):
    ...
  return score
  ```
  - a **dictionary** for the **search space**:
  ```python
  search_space = {"hyperparamter": [...]}
  ```
  - a **dictionary** for the **search-config** that adds both together:
  ```python
  search_config = {model: search_space}
  ```

The model function is the objective function for the optimization. It returns a score that will be **maximized** during the optimization run. The search space is a dictionary that contains the names of the parameters as dict-keys and the list of elements that can be searched during the optimization as dict-values.

?>  Together the model and the search space create the search-config.

<br>

## Create the objective function

The objective function contains the enire model and its evaluation.
The function receives 3 positional arguments:
  - <b>para</b> : (dictionary) This defines what part of the model-function should be optimized
  - <b>X</b> : (numpy array) Training features
  - <b>y</b> : (numpy array) Training target

Via the positional argument **para** you can choose the parameters in the search space. Hyperactive will access the search space during the optimization and try out different positions in the lists.
The function should return some kind of metric that will be <b>maximized</b> during the search.

The finished model should like similar to this:

```python
'''Here you want to optimize the number of estimators of the boosted decition tree
to get the maximum score'''
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

def model(para, X, y):
    gbc = GradientBoostingClassifier(
        '''just put in para["n_estimators"] instead of a value'''
        n_estimators=para["n_estimators"],
    )
    scores = cross_val_score(gbc, X, y, cv=3)

    '''the function must return a metric that will be maximized'''
    return scores.mean()
```



<br>

## Create the search space

The search space is a dictionary that **defines the parameters** and values that will be searched during the optimization run. The keys in the search space **must be the same** as the keys in "para" in the objective function. The values of the search space dictionary are the lists of elements you want to search through. The search space for the model example above could look like this:

```python
search_space = {
    "n_estimators": [50, 100, 150, 200],
}
```

This will search through these four values to find the best one.
You can make the list as long or short as you want:

```python
search_space = {
    "n_estimators": range(10, 200, 10),
}
```

<br>

## Choose an optimizer

Your decision to use a specific optimizer should be based on the time it takes to evaluate a model and if you already have a start point. Try to stick to the following <b>guidelines</b>, when choosing an optimizer:
- only use local or mcmc optimizers, if you have a <b>good start point</b>
- random optimizers are a good way to <b>start exploring</b> the search space
- the majority of the <b>iteration-time</b> should be the <b>evaluation-time</b> of the model

?>  If you want to learn more about the different optimization strategies, check out the corresponding chapters for [local](./optimizers/local_search)-, [random](./optimizers/random_methods)-, [markov-chain-monte-carlo](./optimizers/mcmc)-, [population](./optimizers/population_methods)- and [sequential](./optimizers/sequential_methods)-optimization.

<br>

## Number of iterations

The number of iterations should be low for your first optimization to get to know the iteration-time.
For the <b>iteration-time</b> you should take the following effects into account:
- A <b>k-fold-crossvalidation</b> increases evaluation-time like training on k-1 times on the training data
- Some optimizers will do (and need) <b>multiple evaluations</b> per iteration:
  - Particle-swarm-optimization
  - Evoluion strategy
  - Parallel Tempering
- The <b>complexity</b> of the machine-/deep-learning models will heavily influence the evaluation- and therefore iteration-time.
- The <b>number of epochs</b> should probably be kept low. You just want to compare different types of models. Retrain the best model afterwards with more epochs.

?>  Just start with a small number of iterations (~10) and continue from there.
