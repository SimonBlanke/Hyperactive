import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from hyperactive import Hyperactive

data = load_breast_cancer()
X, y = data.data, data.target


def model(para, X, y):
    model = Sequential()
    model.add(Dense(para["layer0"], activation="relu"))
    model.add(Dropout(para["dropout0"]))
    model.add(Dense(para["layer1"], activation="relu"))
    model.add(Dropout(para["dropout1"]))
    model.add(Dense(1, activation="linear"))

    model.compile(
        loss="categorical_crossentropy", optimizer=RMSprop(), metrics=["accuracy"]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    model.fit(X_train, y_train, batch_size=128, epochs=10, verbose=1)
    score = model.evaluate(X_test, y_test, verbose=0)

    return score, model


# this defines the model and hyperparameter search space
search_config = {
    model: {
        "layer0": range(10, 301, 5),
        "layer1": range(10, 301, 5),
        "dropout0": np.arange(0.1, 1, 0.1),
        "dropout1": np.arange(0.1, 1, 0.1),
    }
}


opt = Hyperactive(search_config, n_iter=10)

# search best hyperparameter for given data
opt.fit(X, y)
