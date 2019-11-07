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

model = applications.VGG19(weights="imagenet", include_top=False)

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


search_config = {
    cnn: {"Dense.0": range(100, 1000, 100), "Dropout.0": np.arange(0.1, 0.9, 0.1)}
}

opt = Hyperactive(search_config, n_iter=5)
opt.search(X_train, y_train)
