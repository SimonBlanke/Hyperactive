# Meta-Learn
Collect and use meta-data of machine learning datasets to reduce search time for hyperparameter. Meta-Learn collects data about model- and dataset-properties to train a regressor (the score of the model being the target).


## Installation
```console
pip install meta-learn
```

## Examples
If you want to search for the best model:
```python
from sklearn.datasets import load_iris
from metalearn import MetaLearn

iris_data = load_iris()
X_train = iris_data.data
y_train = iris_data.target

metalearn = MetaLearn(search_config, metric='accuracy')
metalearn.search(X_train, y_train)

model = metalearn.model
score = metalearn.score
```
