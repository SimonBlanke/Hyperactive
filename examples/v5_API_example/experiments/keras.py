from tensorflow import keras
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from hyperactive.base import BaseExperiment


X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)


class KerasMultiLayerPerceptron(BaseExperiment):
    """
    A class for creating and evaluating a Keras-based Multi-Layer Perceptron (MLP) model.

    This class inherits from BaseExperiment and is designed to build a simple MLP
    using Keras, compile it with the Adam optimizer, and train it on the provided
    training data. The model consists of one hidden dense layer with configurable
    size and activation function, followed by an output layer with a sigmoid
    activation for binary classification.

    Attributes:
        X_train (array-like): Training feature data.
        X_val (array-like): Validation feature data.
        y_train (array-like): Training target data.
        y_val (array-like): Validation target data.

    Methods:
        _score(**params): Builds, compiles, and trains the MLP model using the
        specified parameters for the hidden layer, and returns the validation
        accuracy.
    """

    def __init__(self, X_train, X_val, y_train, y_val):
        super().__init__()

        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val

    def _score(self, **params):
        dense_layer_0 = params["dense_layer_0"]
        activation_layer_0 = params["activation_layer_0"]

        model = keras.Sequential(
            [
                keras.layers.Dense(
                    dense_layer_0,
                    activation=activation_layer_0,
                    input_shape=(20,),
                ),
                keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.01),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(
            self.X_train,
            self.y_train,
            batch_size=32,
            epochs=10,
            validation_data=(self.X_val, self.y_val),
        )
        return model.evaluate(X_val, y_val)[1]
