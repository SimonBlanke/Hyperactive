## Hyperactive

> **Class:** hyperactive.Hyperactive(X, y, memory=True, random_state=1, verbosity=3, warnings=False) <br> [[source]](./source/hyperactive_source)

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

 - ``memory``**:** string or bool, optional (default: "long")

    Collects and reuses evaluation information during optimization. Hyperactive will check if a position has been encountered before. If the position is new it and its score will be saved for later use. If the position has been encountered before the saved score is used instead of reevaluating the model. It is possible to use "short term"-memory or "long term"-memory:
    
         - False -> Does not load, save or use any kind of previous evaluation information to save computation time.
         - "short" -> Does not load or save any kind of previous evaluation information. But temporarily saves and uses evaluation information during optimization without writing it to disk.
         - "long" -> Searches and loads information about previous evaluation information

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

<br>

---

> **Method:** search(search_config, n_iter=10, max_time=None, optimizer='RandomSearch', n_jobs=1) <br> [[source]](./source/search_source)

Starts the optimization run.

#### Parameters

 - ``search_config``**:** dictionary

    Defines the search space and links it to the objective function. 
    
    The objective function takes three positional arguments (para, X, y) and returns a score that will be maximized during the optimization run. The following snippet is an example for an objective function with an sklearn machine learning model in it:

    ```python
    def model(para, X, y):
        gbc = GradientBoostingClassifier(
            n_estimators=para["n_estimators"],
        )
        scores = cross_val_score(gbc, X, y, cv=3)

        return scores.mean()    
    ```
    The objective function is the key of the ``search_config``-dictionary. The search-space is the value of the ``search_config``-dictionary and defines, which parameters to search through during the optimization:
    ```python
    search_config = {
        model: {
            "n_estimators": range(10, 100, 10),
        }
    }
    ```
     The values within the search-space must be lists or numpy arrays.
         
 - ``n_iter``**:** int, optional (default: 10)

    Number of iterations to perform during the optimization run.

 - ``max_time``**:** float, optional (default: None)

    Maximum time in hours to run the optimization. Hyperactive will run the number of iterations from ``n_iter`` and will check after each iteration if the ``max_time`` has been reached. This has the following consequences:
    
     - If the number of iterations has been done the optimization will stop even if ``max_time`` has not been reached.
     - ``max_time`` will not interrupt a model evaluation. It will check if ``max_time`` has been reached in between optimiz ation iterations.
    
 - ``optimizer``**:** string or dictionary, optional (default: "RandomSearch")
 
    Chooses the optimization strategy that is used during the run. 
    You can choose the optimizer two ways:
    
     - by passing a string (like: "StochasticHillClimbing") to ``optimizer``. This chooses the optimizer with its default parameters (like: epsilon=0.03, p_down=1, ...)
     - by passing a nested dictionary (like: {"StochasticHillClimbing": {"epsilon": 0.03, "p_down": 1}}) to ``optimizer``. This chooses the optimizer with the selected parameters.

    List of strings that can be passed to ``optimizer``:

        - "HillClimbing"
        - "StochasticHillClimbing"
        - "TabuSearch"
        - "RandomSearch"
        - "RandomRestartHillClimbing"
        - "RandomAnnealing"
        - "SimulatedAnnealing",
        - "StochasticTunneling"
        - "ParallelTempering"
        - "ParticleSwarm"
        - "EvolutionStrategy"
        - "Bayesian"

    Table of available optimization parameters:

| Argument | type | default |
|---|---|---|
| epsilon | float | 0.03 |
| climb_dist | object | numpy.random.normal |
| n_neighbours | int | 1 |
| p_down | float | 1 |
| tabu_memory | int | 10 |
| n_restarts | int | 10 |
| epsilon_mod | float | 33 |
| annealing_rate | float | 0.99 |
| start_temp | float | 1 |
| gamma | float | 0.5 |
| system_temperatures | list | [0.1, 0.25, 0.5, 0.75, 1, 2.5, 5, 7.5, 10] |
| n_swaps | int | 10 |
| n_particles | int | 10 |
| inertia | float | 0.5 |
| cognitive_weight | float | 0.5 |
| social_weight | float | 0.5 |
| individuals | int | 10 |
| mutation_rate | float | 0.7 |
| crossover_rate | float | 0.3 |
| kernel | object | sklearn.gaussian_process.kernels.Matern(nu=2.5) |


 - ``n_jobs``**:** int, optional (default: 1)

    Number of jobs to run.
    
    
 - ``init_config``**:** dictionary, optional (default: None)
  
     Initialization for the optimization run. 
     Similar to the 'search_config'-parameter the objective function is the key of the dictionary. But the value is an additional dictionary that defines what kind of position initialization should be done. Currently there are three possibilities to initialize the optimization:
     
      - By random:
      
        ```python
        init_config=None
        ```
      - At a given possition:
     
        ```python
        init_config = {
            model: {"n_estimators": [90], "max_depth": [2], "min_samples_split": [5]}
        }
        ```
      - By scatter initialization:
       
        ```python
        init_config = {
            model: {"scatter_init": 10}
        }
        ```

---

> **Attributes:**
  - ``results``**:** dict
    - keys: model-functions
    - values: best parameters
    
  - ``best_scores``**:** dict
    - keys: model-functions
    - values: best scores
    
  - ``eval_times``**:** dict
    - keys: model-functions
    - values: evaluation times of the model
    
  - ``opt_times``**:** dict
    - keys: model-functions
    - values: optimization times of each iteration
    
