"""
RandomSampler Example - Random Search

The RandomSampler performs pure random sampling from the parameter space.
It serves as a baseline and is surprisingly effective for many problems,
especially when the parameter space is high-dimensional or when you have 
limited computational budget.

Characteristics:
- No learning from previous trials
- Uniform sampling from parameter distributions
- Excellent baseline for comparison
- Works well in high-dimensional spaces
- Embarrassingly parallel
- Good when objective function is noisy
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

from hyperactive.experiment.integrations import SklearnCvExperiment
from hyperactive.opt.optuna import RandomSampler


def objective_function_analysis():
    """Demonstrate when random sampling is effective."""
    # When Random Sampling Works Well:
    # 1. High-dimensional parameter spaces (curse of dimensionality)
    # 2. Noisy objective functions
    # 3. Limited computational budget
    # 4. As a baseline for comparison
    # 5. When parallel evaluation is important
    # 6. Uniform exploration is desired


def main():
    # === RandomSampler Example ===
    # Pure Random Search - Uniform Parameter Space Exploration
    
    objective_function_analysis()
    
    # Load dataset - using digits for a more challenging problem
    X, y = load_digits(return_X_y=True)
    print(f"Dataset: Handwritten digits ({X.shape[0]} samples, {X.shape[1]} features)")
    
    # Create experiment
    estimator = SVC(random_state=42)
    experiment = SklearnCvExperiment(estimator=estimator, X=X, y=y, cv=3)
    
    # Define search space - SVM hyperparameters
    param_space = {
        "C": (0.001, 1000),              # Regularization - log scale would be better
        "gamma": (1e-6, 1e2),            # RBF kernel parameter  
        "kernel": ["rbf", "poly", "sigmoid"],  # Kernel type
        "degree": (2, 5),                # Polynomial degree (only for poly kernel)
        "coef0": (0.0, 10.0),           # Independent term (poly/sigmoid)
    }
    
    # Search Space:
    # for param, space in param_space.items():
    #   print(f"  {param}: {space}")
    
    # Configure RandomSampler
    optimizer = RandomSampler(
        param_space=param_space,
        n_trials=30,  # More trials to show random behavior
        random_state=42,  # For reproducible random sampling
        experiment=experiment
    )
    
    # RandomSampler Configuration:
    # n_trials: configured above
    # random_state: set for reproducibility
    # No learning parameters - pure random sampling
    
    # Run optimization
    # Running random search...
    best_params = optimizer.run()
    
    # Results
    print("\n=== Results ===")
    print(f"Best parameters: {best_params}")
    print(f"Best score: {optimizer.best_score_:.4f}")
    print()
    
    # Analysis of Random Sampling behavior:
    # ✓ Each trial is independent - no learning from history
    # ✓ Uniform coverage of parameter space
    # ✓ No convergence issues or local optima concerns
    # ✓ Embarrassingly parallel - can run trials simultaneously
    # ✓ Works equally well for continuous, discrete, and categorical parameters
    
    # Comparison with Other Methods:
    # vs Grid Search:
    #   + Better coverage in high dimensions
    #   + More efficient for continuous parameters
    #   - No systematic coverage guarantee
    #
    # vs Bayesian Optimization (TPE, GP):
    #   + No assumptions about objective function smoothness
    #   + Works well with noisy objectives
    #   + No risk of model misspecification
    #   - No exploitation of promising regions
    #   - May waste trials on clearly bad regions
    
    # Practical Usage:
    # • Use as baseline to validate more sophisticated methods
    # • Good first choice when objective is very noisy
    # • Ideal for parallel optimization setups
    # • Consider for high-dimensional problems (>10 parameters)
    # • Use with log-uniform distributions for scale-sensitive parameters
    
    return best_params, optimizer.best_score_


if __name__ == "__main__":
    best_params, best_score = main()