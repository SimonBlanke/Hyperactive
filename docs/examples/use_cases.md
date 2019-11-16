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


opt = Hyperactive(search_config, n_iter=100)
opt.search(X, y)
```

## Sklearn Pipeline

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


opt = Hyperactive(search_config, n_iter=100)
opt.search(X, y)
```

## Stacking

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from mlxtend.classifier import EnsembleVoteClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from hyperactive import Hyperactive

data = load_breast_cancer()
X, y = data.data, data.target


def model(para, X, y):
    gbc = GradientBoostingClassifier(
        n_estimators=para["n_estimators"], max_depth=para["max_depth"]
    )
    mlp = MLPClassifier(hidden_layer_sizes=para["hidden_layer_sizes"])
    svc = SVC(gamma="auto", probability=True)

    eclf = EnsembleVoteClassifier(
        clfs=[gbc, mlp, svc], weights=[2, 1, 1], voting="soft"
    )

    scores = cross_val_score(eclf, X, y, cv=3)

    return scores.mean()


search_config = {
    model: {
        "n_estimators": range(10, 100, 10),
        "max_depth": range(2, 12),
        "hidden_layer_sizes": (range(10, 100, 10),),
    }
}


opt = Hyperactive(search_config, n_iter=30)
opt.search(X, y)
```

## Transfer Learning

```python
import numpy as np
from keras.models import Sequential
from keras import applications
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Activation
from keras.datasets import cifar10
from keras.utils import to_categorical

from hyperactive import Hyperactive

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = applications.VGG19(weights = "imagenet", include_top=False)

for layer in model.layers[:5]:
    layer.trainable = False

def cnn(para, X_train, y_train):
    model = Sequential()

    model.add(Flatten())
    model.add(Dense(para["Dense.0"]))
    model.add(Activation("relu"))
    model.add(Dropout(para["Dropout.0"]))
    model.add(Dense(10))
    model.add(Activation("softmax"))

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    model.fit(X_train, y_train, epochs=25, batch_size=128)

    loss, score = model.evaluate(x=X_test, y=y_test)

    return score


search_config = {cnn: {"Dense.0": range(100, 1000, 100), "Dropout.0": np.arange(0.1, 0.9, 0.1)}}

opt = Hyperactive(search_config, n_iter=5)
opt.search(X_train, y_train)
```

## Neural Architecture Search

```python
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Activation, Dropout
from keras.datasets import cifar10
from keras.utils import to_categorical

from hyperactive import Hyperactive

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


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


def cnn(para, X_train, y_train):
    model = Sequential()
    model.add(
        Conv2D(para["filters.0"], (3, 3), padding="same", input_shape=X_train.shape[1:])
    )
    model.add(Activation("relu"))
    model.add(Conv2D(para["filters.0"], (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(para["filters.0"], (3, 3), padding="same"))
    model.add(Activation("relu"))
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

    loss, score = model.evaluate(x=X_test, y=y_test)

    return score


search_config = {
    cnn: {
        "conv_layer.0": [conv1, conv2, conv3],
        "filters.0": [16, 32, 64, 128],
        "neurons.0": range(100, 1000, 100),
    }
}

opt = Hyperactive(search_config, n_iter=5)
opt.search(X_train, y_train)
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
model_pretrained.add(
    Conv2D(64, (3, 3), padding="same", input_shape=X_train.shape[1:])
)
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

for i in range(n_layers-8):
    model_pretrained.pop()

for layer in model_pretrained.layers:
    layer.trainable = False

print(model_pretrained.summary())


def cnn(para, X_train, y_train):
    '''
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
    '''
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

    loss, score = model.evaluate(x=X_test, y=y_test)

    return score


search_config = {
    cnn: {
        "conv_layer.0": [conv1, conv2, conv3],
        "neurons.0": range(100, 1000, 100),
    }
}

opt = Hyperactive(search_config, n_iter=5)
opt.search(X_train, y_train)
```
