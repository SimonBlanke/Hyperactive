## Distribution

If the model training does not use all CPU cores, you can start multiple optimizations in <b>parallel</b> by increasing the number of jobs 'n_jobs'. This can make sense if you want to increase the chance of finding the optimal solution or optimize different models at the same time. The parallelization is done by the Multiprocessing-package.

It is also possible to distribute the model training by using the [Ray-package](https://github.com/ray-project/ray). Ray is a powerful framework for building and running distributed applications. Ray can be used with Hyperactive by just importing and initializing Ray. Hyperactive automatically detects this initialization and will use Ray instead of Multiprocessing. You can set the number of jobs 'n_jobs' like before, while passing the ray-specific parameters (like num_cpus, num_gpus, ...) to ray.init().

?>  If you want to learn more about it check out the [distribution-examples](./examples/distribution) and give it a try.




To run multiple optimizations in parallel you can create a search-config with multiple models and search spaces:

```python
'''this also requires to set the n_jobs to 3'''
search_config = {
    model0: search_space0,
    model1: search_space1,
    model2: search_space2,
}
```

<br>


## Optimization Parameters

Table of available optimization parameters:

| Argument | type | default |
|---|---|---|
| epsilon | float | 0.05 |
| distribution | object | normal |
| n_neighbours | int | 1 |
| p_down | float | 0.3 |
| tabu_memory | int | 3 |
| n_restarts | int | 10 |
| annealing_rate | float | 0.99 |
| start_temp | float | 1 |
| gamma | float | 0.5 |
| system_temperatures | list | [0.1, 1, 10, 100] |
| n_swaps | int | 10 |
| n_particles | int | 10 |
| inertia | float | 0.5 |
| cognitive_weight | float | 0.5 |
| social_weight | float | 0.5 |
| individuals | int | 10 |
| mutation_rate | float | 0.7 |
| crossover_rate | float | 0.3 |
| warm_start_smbo  |  bool |  False |
|  xi |  float | 0.01  |
| kernel | object | sklearn.gaussian_process.kernels.Matern(nu=2.5) |
|  start_up_evals |  int |  10 |
|  gamme_tpe |  float | 0.3  |



<br>

## Position Initialization

**Scatter-Initialization**

This technique was inspired by the 'Hyperband Optimization' and aims to find a good initial position for the optimization. It does so by evaluating n random positions with a training subset of 1/n the size of the original dataset. The position that achieves the best score is used as the starting position for the optimization.

**Warm-Start**

When a search is finished the warm-start-dictionary for the best position in the hyperparameter search space (and its metric) is printed in the command line (at verbosity>=1). If multiple searches ran in parallel the warm-start-dictionaries are sorted by the best metric in decreasing order. If the start position in the warm-start-dictionary is not within the search space defined in the search_config an error will occure.
