<details open><summary><b>Create the search space</b></summary>
<p>

Since v0.5.0 the search space is created by defining:
  - a function for the model 
  - a parameter dictionary


The function receives 3 arguments:
  - para : This defines what part of the model-function should be optimized
  - X : Training features
  - y : Training target
  
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
</p>
</details>

<details><summary><b>Choose an optimizer</b></summary>
<p>

Your decision to use a specific optimizer should be based on the time it takes to evaluate a model and if you already have a start point. Try to stick to the following <b>guideline</b>, when choosing an optimizer:
- only use local or mcmc optimizers, if you have a <b>good start point</b>
- random optimizers are a good way to <b>start exploring</b> the search space
- the majority of the <b>iteration-time</b> should be the <b>evaluation-time</b> of the model

You can choose an optimizer-class from the list provided in the [API](https://github.com/SimonBlanke/Hyperactive#hyperactive-api).
All optimization techniques are explained in more detail [here](https://github.com/SimonBlanke/Hyperactive/tree/master/hyperactive/optimizers#optimization-techniques). A comparison between the iteration- and evaluation-time for different models can be seen [here](https://github.com/SimonBlanke/Hyperactive/tree/master/hyperactive/model#supported-packages).

</p>
</details>

<details><summary><b>How many iterations?</b></summary>
<p>

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

</p>
</details>

<details><summary><b>Distribution (optional)</b></summary>
<p>
  
You can start multiple optimizations in <b>parallel</b> by increasing the number of jobs. This can make sense if you want to increase the chance of finding the optimal solution or optimize different models at the same time.
  
</p>
</details>

<details><summary><b>Advanced features (optional)</b></summary>
<p>

The [advanced features](https://github.com/SimonBlanke/Hyperactive/tree/master/hyperactive#advanced-features) can be very useful to improve the performance of the optimizers in some situations. The 'memory' is used by default, because it saves you a lot of time.

</p>
</details>
