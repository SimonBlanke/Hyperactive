"""
TPESampler Example - Tree-structured Parzen Estimator

The TPESampler is Optuna's default and most popular Bayesian optimization algorithm.
It uses a Tree-structured Parzen Estimator to model the relationship between
hyperparameters and objective values, making it efficient at finding optimal regions.

Characteristics:
- Bayesian optimization approach
- Good balance of exploration vs exploitation
- Works well with mixed parameter types (continuous, discrete, categorical)
- Efficient for moderate-dimensional problems
- Default choice for most hyperparameter optimization tasks
"""

from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier

from hyperactive.experiment.integrations import SklearnCvExperiment
from hyperactive.opt.optuna import TPEOptimizer


def main():
    # === TPESampler Example ===
    # Tree-structured Parzen Estimator - Bayesian Optimization

    # Load dataset
    X, y = load_wine(return_X_y=True)
    print(f"Dataset: Wine classification ({X.shape[0]} samples, {X.shape[1]} features)")

    # Create experiment
    estimator = RandomForestClassifier(random_state=42)
    experiment = SklearnCvExperiment(estimator=estimator, X=X, y=y, cv=3)

    # Define search space
    param_space = {
        "n_estimators": (10, 200),           # Continuous integer
        "max_depth": (1, 20),                # Continuous integer
        "min_samples_split": (2, 20),        # Continuous integer
        "min_samples_leaf": (1, 10),         # Continuous integer
        "max_features": ["sqrt", "log2", None],  # Categorical
        "bootstrap": [True, False],          # Categorical boolean
    }

    # Search Space:
    # for param, space in param_space.items():
    #   print(f"  {param}: {space}")

    # Configure TPESampler with warm start
    warm_start_points = [
        {"n_estimators": 100, "max_depth": 10, "min_samples_split": 2,
         "min_samples_leaf": 1, "max_features": "sqrt", "bootstrap": True}
    ]

    optimizer = TPEOptimizer(
        param_space=param_space,
        n_trials=50,
        random_state=42,
        initialize={"warm_start": warm_start_points},
        experiment=experiment,
        n_startup_trials=10,  # Random trials before TPE kicks in
        n_ei_candidates=24    # Number of candidates for expected improvement
    )

    # TPESampler Configuration:
    # n_trials: configured above
    # n_startup_trials: random exploration phase
    # n_ei_candidates: number of expected improvement candidates
    # warm_start: initial point(s) provided

    # Run optimization
    # Running optimization...
    best_params = optimizer.solve()

    # Results
    print("\n=== Results ===")
    print(f"Best parameters: {best_params}")
    print(f"Best score: {optimizer.best_score_:.4f}")
    print()

    # TPE Behavior Analysis:
    # - First 10 trials: Random exploration (n_startup_trials)
    # - Trials 11-50: TPE-guided exploration based on past results
    # - TPE builds probabilistic models of good vs bad parameter regions
    # - Balances exploration of uncertain areas with exploitation of promising regions

    # Parameter Space Exploration:
    # TPESampler effectively explores the joint parameter space by:
    # 1. Modeling P(x|y) - probability of parameters given objective values
    # 2. Using separate models for 'good' and 'bad' performing regions
    # 3. Selecting next points to maximize expected improvement
    # 4. Handling mixed parameter types (continuous, discrete, categorical)

    return best_params, optimizer.best_score_


if __name__ == "__main__":
    best_params, best_score = main()
