## Sklearn Preprocessing

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import GradientBoostingClassifier
from hyperactive import Hyperactive

data = load_breast_cancer()
X, y = data.data, data.target


def pca(X):
    X = PCA(n_components=10).fit_transform(X)

    return X


def none(X):
    return X


def model(para, X, y):
    model = GradientBoostingClassifier(
        n_estimators=para["n_estimators"],
        max_depth=para["max_depth"],
        min_samples_split=para["min_samples_split"],
        min_samples_leaf=para["min_samples_leaf"],
    )

    X_pca = para["decomposition"](X)
    X = np.hstack((X, X_pca))

    X = SelectKBest(f_classif, k=para["k"]).fit_transform(X, y)
    scores = cross_val_score(model, X, y, cv=3)

    return scores.mean()


search_config = {
    model: {
        "decomposition": [pca, none],
        "k": range(2, 30),
        "n_estimators": range(10, 200, 10),
        "max_depth": range(2, 12),
        "min_samples_split": range(2, 12),
        "min_samples_leaf": range(1, 11),
    }
}


opt = Hyperactive(X, y)
opt.search(search_config, n_iter=100)
```

## Sklearn Pipeline

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline

from hyperactive import Hyperactive

data = load_breast_cancer()
X, y = data.data, data.target


def pipeline1(filter_, gbc):
    return Pipeline([("filter_", filter_), ("gbc", gbc)])


def pipeline2(filter_, gbc):
    return gbc


def model(para, X, y):
    gbc = GradientBoostingClassifier(
        n_estimators=para["n_estimators"],
        max_depth=para["max_depth"],
        min_samples_split=para["min_samples_split"],
        min_samples_leaf=para["min_samples_leaf"],
    )
    filter_ = SelectKBest(f_classif, k=para["k"])
    model_ = para["pipeline"](filter_, gbc)

    scores = cross_val_score(model_, X, y, cv=3)

    return scores.mean()


search_config = {
    model: {
        "k": range(2, 30),
        "n_estimators": range(10, 200, 10),
        "max_depth": range(2, 12),
        "min_samples_split": range(2, 12),
        "min_samples_leaf": range(1, 11),
        "pipeline": [pipeline1, pipeline2],
    }
}


opt = Hyperactive(X, y)
opt.search(search_config, n_iter=100)
```

## Stacking

```python
import itertools

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from mlxtend.classifier import StackingClassifier

from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model.ridge import RidgeClassifier

from hyperactive import Hyperactive

data = load_breast_cancer()
X, y = data.data, data.target


gbc = GradientBoostingClassifier()
rfc = RandomForestClassifier()
etc = ExtraTreesClassifier()

mlp = MLPClassifier()
gnb = GaussianNB()
gpc = GaussianProcessClassifier()
dtc = DecisionTreeClassifier()
knn = KNeighborsClassifier()

lr = LogisticRegression()
rc = RidgeClassifier()


def stacking(para, X, y):
    stack_lvl_0 = StackingClassifier(
        classifiers=para["lvl_0"], meta_classifier=para["top"]
    )
    stack_lvl_1 = StackingClassifier(
        classifiers=para["lvl_1"], meta_classifier=stack_lvl_0
    )
    scores = cross_val_score(stack_lvl_1, X, y, cv=3)

    return scores.mean()


def get_combinations(models):
    comb = []
    for i in range(0, len(models) + 1):
        for subset in itertools.permutations(models, i):
            if len(subset) == 0:
                continue
            comb.append(list(subset))
    return comb


top = [lr, dtc, gnb, rc]
models_0 = [gpc, dtc, mlp, gnb, knn]
models_1 = [gbc, rfc, etc]

stack_lvl_0_clfs = get_combinations(models_0)
stack_lvl_1_clfs = get_combinations(models_1)


search_config = {
    stacking: {"lvl_1": stack_lvl_1_clfs, "lvl_0": stack_lvl_0_clfs, "top": top}
}


opt = Hyperactive(X, y)
opt.search(search_config, n_jobs=2, n_iter=150)
```

## Neural Architecture Search

```python
from keras.nns import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Activation, Dropout
from keras.datasets import cifar10
from keras.utils import to_categorical

from hyperactive import Hyperactive

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


def conv1(nn):
    nn.add(Conv2D(32, (3, 3)))
    nn.add(Activation("relu"))
    nn.add(MaxPooling2D(pool_size=(2, 2)))
    return nn


def conv2(nn):
    nn.add(Conv2D(32, (3, 3)))
    nn.add(Activation("relu"))
    return nn


def conv3(nn):
    return nn


def cnn(para, X_train, y_train):
    nn = Sequential()
    nn.add(
        Conv2D(para["filters.0"], (3, 3), padding="same", input_shape=X_train.shape[1:])
    )
    nn.add(Activation("relu"))
    nn.add(Conv2D(para["filters.0"], (3, 3)))
    nn.add(Activation("relu"))
    nn.add(MaxPooling2D(pool_size=(2, 2)))
    nn.add(Dropout(0.25))

    nn.add(Conv2D(para["filters.0"], (3, 3), padding="same"))
    nn.add(Activation("relu"))
    nn = para["conv_layer.0"](nn)
    nn.add(Dropout(0.25))

    nn.add(Flatten())
    nn.add(Dense(para["neurons.0"]))
    nn.add(Activation("relu"))
    nn.add(Dropout(0.5))
    nn.add(Dense(10))
    nn.add(Activation("softmax"))

    nn.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    nn.fit(X_train, y_train, epochs=25, batch_size=128)

    _, score = nn.evaluate(x=X_test, y=y_test)

    return score


search_config = {
    cnn: {
        "conv_layer.0": [conv1, conv2, conv3],
        "filters.0": [16, 32, 64, 128],
        "neurons.0": range(100, 1000, 100),
    }
}

opt = Hyperactive(X_train, y_train)
opt.search(search_config, n_iter=5)
```

## ENAS

```python
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Activation, Dropout
from keras.datasets import cifar10
from keras.utils import to_categorical

from hyperactive import Hyperactive

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


"""
Efficient Neural Architecture Search via Parameter Sharing:
https://arxiv.org/pdf/1802.03268.pdf
"""


def conv1(model):
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    return model


def conv2(model):
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation("relu"))
    return model


def conv3(model):
    return model


model_pretrained = Sequential()
model_pretrained.add(Conv2D(64, (3, 3), padding="same", input_shape=X_train.shape[1:]))
model_pretrained.add(Activation("relu"))
model_pretrained.add(Conv2D(32, (3, 3)))
model_pretrained.add(Activation("relu"))
model_pretrained.add(MaxPooling2D(pool_size=(2, 2)))
model_pretrained.add(Dropout(0.25))

model_pretrained.add(Conv2D(32, (3, 3), padding="same"))
model_pretrained.add(Activation("relu"))
model_pretrained.add(Dropout(0.25))

model_pretrained.add(Flatten())
model_pretrained.add(Dense(200))
model_pretrained.add(Activation("relu"))
model_pretrained.add(Dropout(0.5))
model_pretrained.add(Dense(10))
model_pretrained.add(Activation("softmax"))

model_pretrained.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)
model_pretrained.fit(X_train, y_train, epochs=50, batch_size=128)

n_layers = len(model_pretrained.layers)

for i in range(n_layers - 8):
    model_pretrained.pop()

for layer in model_pretrained.layers:
    layer.trainable = False

print(model_pretrained.summary())


def cnn(para, X_train, y_train):
    """
    model = Sequential()
    model.add(
        Conv2D(64, (3, 3), padding="same", input_shape=X_train.shape[1:])
    )
    model.add(Activation("relu"))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation("relu"))


    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(32, (3, 3), padding="same"))
    model.add(Activation("relu"))
    """
    model = model_pretrained

    model = para["conv_layer.0"](model)
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(para["neurons.0"]))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation("softmax"))

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    model.fit(X_train, y_train, epochs=25, batch_size=128)

    _, score = model.evaluate(x=X_test, y=y_test)

    return score


search_config = {
    cnn: {"conv_layer.0": [conv1, conv2, conv3], "neurons.0": range(100, 1000, 100)}
}

opt = Hyperactive(X_train, y_train)
opt.search(search_config, n_iter=5)
```

## Transfer Learning

```python
import numpy as np
from keras.models import Sequential
from keras import applications
from keras.layers import Dense, Flatten, Dropout, Activation
from keras.datasets import cifar10
from keras.utils import to_categorical

from hyperactive import Hyperactive

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

nn = applications.VGG19(weights="imagenet", include_top=False)

for layer in nn.layers[:5]:
    layer.trainable = False


def cnn(para, X_train, y_train):
    nn = Sequential()

    nn.add(Flatten())
    nn.add(Dense(para["Dense.0"]))
    nn.add(Activation("relu"))
    nn.add(Dropout(para["Dropout.0"]))
    nn.add(Dense(10))
    nn.add(Activation("softmax"))

    nn.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    nn.fit(X_train, y_train, epochs=25, batch_size=128)

    _, score = nn.evaluate(x=X_test, y=y_test)

    return score


search_config = {
    cnn: {"Dense.0": range(100, 1000, 100), "Dropout.0": np.arange(0.1, 0.9, 0.1)}
}

opt = Hyperactive(X_train, y_train)
opt.search(search_config, n_iter=5)
```

## Meta-Optimization

```python
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from hyperactive import Hyperactive

data = load_breast_cancer()
X, y = data.data, data.target


def meta_opt(para, X, y):
    def model(para, X, y):
        model = DecisionTreeClassifier(
            max_depth=para["max_depth"],
            min_samples_split=para["min_samples_split"],
            min_samples_leaf=para["min_samples_leaf"],
        )
        scores = cross_val_score(model, X, y, cv=3)

        return scores.mean()

    search_config = {
        model: {
            "max_depth": range(2, 50),
            "min_samples_split": range(2, 50),
            "min_samples_leaf": range(1, 50),
        }
    }

    opt = Hyperactive(
        search_config,
        optimizer={
            "ParticleSwarm": {
                "inertia": para["inertia"],
                "cognitive_weight": para["cognitive_weight"],
                "social_weight": para["social_weight"],
            }
        },
        verbosity=None,
    )
    opt.search(X, y)

    return opt.score_best


search_config = {
    meta_opt: {
        "inertia": np.arange(0, 1, 0.01),
        "cognitive_weight": np.arange(0, 1, 0.01),
        "social_weight": np.arange(0, 1, 0.01),
    }
}

opt = Hyperactive(X, y)
opt.search(search_config, optimizer="Bayesian", n_iter=50)
```

