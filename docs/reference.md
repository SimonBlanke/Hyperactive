## Hyperactive

> **class:** hyperactive.Hyperactive(X, y, memory=True, random_state=1, verbosity=3, warnings=False) <br> [[source]](./source/hyperactive_source)

Optimization main class.

#### Parameters

 - ``X``**:** array-like or None

    Training input samples used during the optimization process.
    The training data is passed to the ``X`` argument in the objective function during the optimization process.
    You can also pass ``None`` if you want to optimize an objective function that does not contain a machine learning model.

 - ``y``**:** array-like or None

    Training target values used during the optimization process.
    The target values are passed to the ``y`` argument in the objective function during the optimization process.
    You can also pass ``None`` if you want to optimize an objective function that does not contain a machine learning model.

 - ``memory``**:** bool, optional (default: True)

    Collects and reuses evaluation information during optimization.

 - ``random_state``**:** int, optional (default: 1)

    The seed of the pseudo random number generator.

 - ``verbosity``**:** int, optional (default: 3)

    How much information Hyperactive provides during the optimization process:

        - 0 -> No information from Hyperactive.
        - 1 -> Prints scores and positions of best evaluation after optimization.
        - 2 -> Adds progress bar(s) with minimal information.
        - 3 -> Adds current best score and iteration number of current best score to progress bar information.

 - ``warnings``**:** bool, optional (default: False)

    Disables warnings (like deprecation warnings) during optimization:
    Warnings can be very intrusive, since they are often printed during each model evaluation.


> **Method:** search(search_config, n_iter=10, max_time=None, optimizer='RandomSearch', n_jobs=1) <br> [[source]](./source/search_source)

Starts the optimization run.

#### Parameters

 - ``search_config``**:** dictionary

    Defines the search space and links it to the objective function.
    The objective function is the key of the dictionary, while the search space (which is also a dictionary) is the value.
    You can define multiple modeles/search-spaces in the search_config.
    The values within the search space (not search_config) must be lists or numpy arrays.

 - ``n_iter``**:** int, optional (default: 10)

    Number of iterations.

 - ``max_time``**:** float, optional (default: None)

    Maximum time in hours to run the optimization.

 - ``optimizer``**:** string or dict, optional (default: "RandomSearch")

    Optimization strategy used during the run.

 - ``n_jobs``**:** int, optional (default: 1)

    Number of jobs to run.
