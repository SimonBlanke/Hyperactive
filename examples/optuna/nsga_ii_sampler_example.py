"""
NSGAIISampler Example - Multi-objective Optimization with NSGA-II

NSGA-II (Non-dominated Sorting Genetic Algorithm II) is designed for
multi-objective optimization problems where you want to optimize multiple
conflicting objectives simultaneously. It finds a Pareto front of solutions.

Characteristics:
- Multi-objective evolutionary algorithm
- Finds Pareto-optimal solutions (non-dominated set)
- Balances multiple conflicting objectives
- Population-based search with selection pressure
- Elitist approach preserving best solutions
- Crowding distance for diversity preservation

Note: For demonstration, we'll create a multi-objective problem from
a single-objective one by optimizing both performance and model complexity.
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from hyperactive.experiment.integrations import SklearnCvExperiment
from hyperactive.opt.optuna import NSGAIISampler


class MultiObjectiveExperiment:
    """Multi-objective experiment: maximize accuracy, minimize complexity."""

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __call__(self, **params):
        # Create model with parameters
        model = RandomForestClassifier(random_state=42, **params)

        # Objective 1: Maximize accuracy (we'll return negative for minimization)
        scores = cross_val_score(model, self.X, self.y, cv=3)
        accuracy = np.mean(scores)

        # Objective 2: Minimize model complexity (number of parameters)
        # For Random Forest: roughly n_estimators × max_depth × n_features
        complexity = (
            params["n_estimators"] * params.get("max_depth", 10) * self.X.shape[1]
        )

        # NSGA-II minimizes objectives, so we return both as minimization
        # Note: This is a simplified multi-objective setup for demonstration
        return [-accuracy, complexity / 10000]  # Scale complexity for better balance


def nsga_ii_theory():
    """Explain NSGA-II algorithm theory."""
    # NSGA-II Algorithm (Multi-objective Optimization):
    #
    # 1. Core Concepts:
    #    - Pareto Dominance: Solution A dominates B if A is better in all objectives
    #    - Pareto Front: Set of non-dominated solutions
    #    - Trade-offs: Improving one objective may worsen another
    #
    # 2. NSGA-II Process:
    #    - Initialize population randomly
    #    - For each generation:
    #      a) Fast non-dominated sorting (rank solutions by dominance)
    #      b) Crowding distance calculation (preserve diversity)
    #      c) Selection based on rank and crowding distance
    #      d) Crossover and mutation to create offspring
    #
    # 3. Selection Criteria:
    #    - Primary: Non-domination rank (prefer better fronts)
    #    - Secondary: Crowding distance (prefer diverse solutions)
    #    - Elitist: Best solutions always survive
    #
    # 4. Output:
    #    - Set of Pareto-optimal solutions
    #    - User chooses final solution based on preferences


def main():
    # === NSGAIISampler Example ===
    # Multi-objective Optimization with NSGA-II

    nsga_ii_theory()

    # Load dataset
    X, y = load_digits(return_X_y=True)
    print(f"Dataset: Handwritten digits ({X.shape[0]} samples, {X.shape[1]} features)")

    # Create multi-objective experiment
    experiment = MultiObjectiveExperiment(X, y)

    # Multi-objective Problem:
    #   Objective 1: Maximize classification accuracy
    #   Objective 2: Minimize model complexity
    #   → Trade-off between performance and simplicity

    # Define search space
    param_space = {
        "n_estimators": (10, 200),  # Number of trees
        "max_depth": (1, 20),  # Tree depth (complexity)
        "min_samples_split": (2, 20),  # Minimum samples to split
        "min_samples_leaf": (1, 10),  # Minimum samples per leaf
        "max_features": ["sqrt", "log2", None],  # Feature sampling
    }

    # Search Space:
    # for param, space in param_space.items():
    #   print(f"  {param}: {space}")

    # Configure NSGAIISampler
    optimizer = NSGAIISampler(
        param_space=param_space,
        n_trials=50,  # Population evolves over multiple generations
        random_state=42,
        experiment=experiment,
        population_size=20,  # Population size for genetic algorithm
        mutation_prob=0.1,  # Mutation probability
        crossover_prob=0.9,  # Crossover probability
    )

    # NSGAIISampler Configuration:
    # n_trials: configured above
    # population_size: for genetic algorithm
    # mutation_prob: mutation probability
    # crossover_prob: crossover probability
    # Selection: Non-dominated sorting + crowding distance

    # Note: This example demonstrates the interface.
    # In practice, NSGA-II returns multiple Pareto-optimal solutions.
    # For single-objective problems, consider TPE or GP samplers instead.

    # Run optimization
    # Running NSGA-II multi-objective optimization...

    try:
        best_params = optimizer.run()

        # Results
        print("\n=== Results ===")
        print(f"Best parameters: {best_params}")
        print(f"Best score: {optimizer.best_score_:.4f}")
        print()

        # NSGA-II typically returns multiple solutions along Pareto front:
        #  High accuracy, high complexity models
        #  Medium accuracy, medium complexity models
        #  Lower accuracy, low complexity models
        #  User selects based on preferences/constraints

    except Exception as e:
        print(f"Multi-objective optimization example: {e}")
        print("Note: This demonstrates the interface for multi-objective problems.")
        return None, None

    # NSGA-II Evolution Process:
    #
    # Generation 1: Random initialization
    #  Diverse population across parameter space
    #  Wide range of accuracy/complexity trade-offs

    # Generations 2-N: Evolutionary improvement
    #  Non-dominated sorting identifies best fronts
    #  Crowding distance maintains solution diversity
    #  Crossover combines good solutions
    #  Mutation explores new parameter regions

    # Final Population: Pareto front approximation
    #  Multiple non-dominated solutions
    #  Represents optimal trade-offs
    #  User chooses based on domain requirements

    # Key Advantages:
    #  Handles multiple conflicting objectives naturally
    #  Finds diverse set of optimal trade-offs
    #  No need to specify objective weights a priori
    #  Provides insight into objective relationships
    #  Robust to objective scaling differences

    # Best Use Cases:
    #  True multi-objective problems (accuracy vs speed, cost vs quality)
    #  When trade-offs between objectives are important
    #  Robustness analysis with multiple criteria
    #  When single objective formulation is unclear

    # Limitations:
    #  More complex than single-objective methods
    #  Requires more evaluations (population-based)
    #  May be overkill for single-objective problems
    #  Final solution selection still required

    # When to Use NSGA-II vs Single-objective Methods:
    # Use NSGA-II when:
    #    Multiple objectives genuinely conflict
    #    Trade-off analysis is valuable
    #    Objective weights are unknown
    #
    # Use TPE/GP when:
    #    Single clear objective
    #    Computational budget is limited
    #    Faster convergence needed

    if "best_params" in locals():
        return best_params, optimizer.best_score_
    else:
        return None, None


if __name__ == "__main__":
    best_params, best_score = main()
