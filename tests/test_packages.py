# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import cross_val_score
from hyperactive import Hyperactive

data = load_breast_cancer()
X, y = data.data, data.target


def test_meta_learn():
    from sklearn.tree import DecisionTreeClassifier

    def model(para, X_train, y_train):
        model = DecisionTreeClassifier(
            max_depth=para["max_depth"],
            min_samples_split=para["min_samples_split"],
            min_samples_leaf=para["min_samples_leaf"],
        )
        scores = cross_val_score(model, X_train, y_train, cv=3)

        return scores.mean()

    search_config = {
        model: {
            "max_depth": range(1, 21),
            "min_samples_split": range(2, 21),
            "min_samples_leaf": range(1, 21),
        }
    }

    opt = Hyperactive(search_config, memory="long")
    opt.search(X, y)


def test_sklearn():
    from sklearn.tree import DecisionTreeClassifier

    def model(para, X_train, y_train):
        model = DecisionTreeClassifier(
            criterion=para["criterion"],
            max_depth=para["max_depth"],
            min_samples_split=para["min_samples_split"],
            min_samples_leaf=para["min_samples_leaf"],
        )
        scores = cross_val_score(model, X_train, y_train, cv=3)

        return scores.mean()

    search_config = {
        model: {
            "criterion": ["gini", "entropy"],
            "max_depth": range(1, 21),
            "min_samples_split": range(2, 21),
            "min_samples_leaf": range(1, 21),
        }
    }

    opt = Hyperactive(search_config)
    opt.search(X, y)
    # opt.predict(X)
    # opt.score(X, y)


def test_xgboost():
    from xgboost import XGBClassifier

    def model(para, X_train, y_train):
        model = XGBClassifier(
            n_estimators=para["n_estimators"], max_depth=para["max_depth"]
        )
        scores = cross_val_score(model, X_train, y_train, cv=3)

        return scores.mean()

    search_config = {model: {"n_estimators": range(2, 20), "max_depth": range(1, 11)}}

    opt = Hyperactive(search_config)
    opt.search(X, y)
    # opt.predict(X)
    # opt.score(X, y)


def test_lightgbm():
    from lightgbm import LGBMClassifier

    def model(para, X_train, y_train):
        model = LGBMClassifier(
            num_leaves=para["num_leaves"], learning_rate=para["learning_rate"]
        )
        scores = cross_val_score(model, X_train, y_train, cv=3)

        return scores.mean()

    search_config = {
        model: {
            "num_leaves": range(2, 20),
            "learning_rate": [0.001, 0.005, 00.01, 0.05, 0.1, 0.5, 1],
        }
    }

    opt = Hyperactive(search_config)
    opt.search(X, y)
    # opt.predict(X)
    # opt.score(X, y)


def test_catboost():
    from catboost import CatBoostClassifier

    def model(para, X_train, y_train):
        model = CatBoostClassifier(
            iterations=para["iterations"],
            depth=para["depth"],
            learning_rate=para["learning_rate"],
        )
        scores = cross_val_score(model, X_train, y_train, cv=3)

        return scores.mean()

    search_config = {
        model: {
            "iterations": [1],
            "depth": range(2, 10),
            "learning_rate": [0.001, 0.005, 00.01, 0.05, 0.1, 0.5, 1],
        }
    }

    opt = Hyperactive(search_config)
    opt.search(X, y)
    # opt.predict(X)
    # opt.score(X, y)


def test_keras():
    from keras.models import Sequential
    from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
    from keras.datasets import cifar10
    from keras.utils import to_categorical

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    X_train = X_train[0:1000]
    y_train = y_train[0:1000]

    X_test = X_train[0:1000]
    y_test = y_train[0:1000]

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    def cnn(para, X_train, y_train):
        model = Sequential()

        model.add(
            Conv2D(
                filters=para["filters.0"],
                kernel_size=para["kernel_size.0"],
                activation="relu",
            )
        )
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(10, activation="softmax"))

        model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )
        model.fit(X_train, y_train, epochs=1)

        _, score = model.evaluate(x=X_test, y=y_test)

        return score

    search_config = {cnn: {"filters.0": [32, 64], "kernel_size.0": [3, 4]}}

    opt = Hyperactive(search_config)
    opt.search(X_train, y_train)
