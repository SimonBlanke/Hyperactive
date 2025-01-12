from tensorflow import keras
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from hyperactive import BaseExperiment


X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)


class KerasMultiLayerPerceptron(BaseExperiment):
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
