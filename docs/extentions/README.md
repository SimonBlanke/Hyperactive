## Optimization Extensions

The features listed below can be activated during the instantiation of the optimizer ([see API](https://github.com/SimonBlanke/hyperactive#hyperactive-api)) and works with every optimizer in the hyperactive package.


## Position initialization

#### [Scatter-Initialization](https://github.com/SimonBlanke/Hyperactive/blob/master/examples/example_scatter_init.py)
This technique was inspired by the 'Hyperband Optimization' and aims to find a good initial position for the optimization. It does so by evaluating n random positions with a training subset of 1/n the size of the original dataset. The position that achieves the best score is used as the starting position for the optimization.


#### [Warm-Start](https://github.com/SimonBlanke/Hyperactive/blob/master/examples/example_warm_start_sklearn.py)

When a search is finished the warm-start-dictionary for the best position in the hyperparameter search space (and its metric) is printed in the command line (at verbosity=1). If multiple searches ran in parallel the warm-start-dictionaries are sorted by the best metric in decreasing order. If the start position in the warm-start-dictionary is not within the search space defined in the search_config an error will occure.

<details><summary>Warm-start example for sklearn model:</summary>
<p>

```python
start_point = {
    "sklearn.ensemble.RandomForestClassifier.0": {
        "n_estimators": [30],
        "max_depth": [6],
        "criterion": ["entropy"],
        "min_samples_split": [12],
        "min_samples_leaf": [16],
    },
    "sklearn.ensemble.RandomForestClassifier.1": {
        "n_estimators": [50],
        "max_depth": [3],
        "criterion": ["entropy"],
    },
}
```

</p>
</details>


<details><summary>Warm-start example for keras model (cnn):</summary>
<p>

```python
start_point = {
    "keras.compile.0": {"loss": ["categorical_crossentropy"], "optimizer": ["adam"]},
    "keras.fit.0": {"epochs": [3], "batch_size": [500], "verbose": [0]},
    "keras.layers.Conv2D.1": {
        "filters": [64],
        "kernel_size": [3],
        "activation": ["relu"],
        "input_shape": [(28, 28, 1)],
    },
    "keras.layers.MaxPooling2D.2": {"pool_size": [(2, 2)]},
    "keras.layers.Conv2D.3": {
        "filters": [32],
        "kernel_size": [3],
        "activation": ["relu"],
        "input_shape": [(28, 28, 1)],
    },
    "keras.layers.MaxPooling2D.4": {"pool_size": [(2, 2)]},
    "keras.layers.Conv2D.5": {
        "filters": [32],
        "kernel_size": [3],
        "activation": ["relu"],
        "input_shape": [(28, 28, 1)],
    },
    "keras.layers.MaxPooling2D.6": {"pool_size": [(2, 2)]},
    "keras.layers.Flatten.7": {},
    "keras.layers.Dense.8": {"units": [50], "activation": ["softmax"]},
    "keras.layers.Dropout.9": {"rate": [0.4]},
"keras.layers.Dense.10": {"units": [10], "activation": ["softmax"]},
}
```

</p>
</details>



## Resources allocation

#### [Memory](https://github.com/SimonBlanke/Hyperactive/blob/master/examples/example_memory.py)
After the evaluation of a model the position (in the hyperparameter search dictionary) and the cross-validation score are written to a dictionary. If the optimizer tries to evaluate this position again it can quickly lookup if a score for this position is present and use it instead of going through the extensive training and prediction process.
