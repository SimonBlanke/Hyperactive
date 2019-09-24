## Create the search space

Since v0.5.0 the search space is created by defining:
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
