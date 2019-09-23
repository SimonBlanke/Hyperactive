## Optimization Extensions

The features listed below can be activated during the instantiation of the optimizer ([see API](https://github.com/SimonBlanke/hyperactive#hyperactive-api)) and works with every optimizer in the hyperactive package.


## Position initialization

#### [Scatter-Initialization](https://github.com/SimonBlanke/Hyperactive/blob/master/examples/example_scatter_init.py)
This technique was inspired by the 'Hyperband Optimization' and aims to find a good initial position for the optimization. It does so by evaluating n random positions with a training subset of 1/n the size of the original dataset. The position that achieves the best score is used as the starting position for the optimization.


#### [Warm-Start](https://github.com/SimonBlanke/Hyperactive/blob/master/examples/example_warm_start_sklearn.py)

When a search is finished the warm-start-dictionary for the best position in the hyperparameter search space (and its metric) is printed in the command line (at verbosity=1). If multiple searches ran in parallel the warm-start-dictionaries are sorted by the best metric in decreasing order. If the start position in the warm-start-dictionary is not within the search space defined in the search_config an error will occure.


## Resources allocation

#### [Memory](https://github.com/SimonBlanke/Hyperactive/blob/master/examples/example_memory.py)
After the evaluation of a model the position (in the hyperparameter search dictionary) and the cross-validation score are written to a dictionary. If the optimizer tries to evaluate this position again it can quickly lookup if a score for this position is present and use it instead of going through the extensive training and prediction process.
