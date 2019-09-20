# Hyperactive


## Installation
PyPi always has the newest version of hyperactive:
```console
pip install hyperactive
```

## Minimal example

```og
!main

class Oglang<T>
  Foo T
  GetFoo: T -> @Foo

main ->
  foo := Oglang<int>
    Foo: 42

  foo.GetFoo()
```


```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from hyperactive import Hyperactive

data = load_breast_cancer()
X, y = data.data, data.target

def model(para, X, y):
    model = GradientBoostingClassifier(n_estimators=para['n_estimators'])
    scores = cross_val_score(model, X, y)

    return scores.mean()

search_config = {
    model: {'n_estimators': range(10, 200, 10)}
}

opt = Hyperactive(search_config)
opt.fit(X, y)
```
