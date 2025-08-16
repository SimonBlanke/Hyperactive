"""
NSGAIIISampler Example - Many-objective Optimization with NSGA-III

NSGA-III is an extension of NSGA-II specifically designed for many-objective
optimization problems (typically 3+ objectives). It uses reference points
to maintain diversity and selection pressure in high-dimensional objective spaces.

Characteristics:
- Many-objective evolutionary algorithm (3+ objectives)
- Reference point-based selection mechanism
- Better performance than NSGA-II for many objectives
- Maintains diversity through structured reference points
- Elitist approach with improved selection pressure
- Population-based search with normalization

Note: For demonstration, we'll create a many-objective problem optimizing
accuracy, complexity, training time, and model interpretability.
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import time

from hyperactive.experiment.integrations import SklearnCvExperiment
from hyperactive.opt.optuna import NSGAIIISampler


class ManyObjectiveExperiment:
    """Many-objective experiment: optimize multiple conflicting goals."""

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __call__(self, **params):
        # Create model with parameters
        model = DecisionTreeClassifier(random_state=42, **params)

        # Objective 1: Maximize accuracy (return negative for minimization)
        start_time = time.time()
        scores = cross_val_score(model, self.X, self.y, cv=3)
        training_time = time.time() - start_time
        accuracy = np.mean(scores)

        # Objective 2: Minimize model complexity (tree depth)
        complexity = params.get("max_depth", 20)

        # Objective 3: Minimize training time
        time_objective = training_time

        # Objective 4: Maximize interpretability (minimize tree size)
        # Approximate tree size based on parameters
        max_leaf_nodes = params.get("max_leaf_nodes", 100)
        interpretability = max_leaf_nodes / 100.0  # Normalized

        # Return all objectives for minimization (negative accuracy for maximization)
        return [
            -accuracy,  # Minimize negative accuracy (maximize accuracy)
            complexity / 20.0,  # Minimize complexity (normalized)
            time_objective,  # Minimize training time
            interpretability,  # Minimize tree size (maximize interpretability)
        ]


def nsga_iii_theory():
    """Explain NSGA-III algorithm theory."""
    # NSGA-III Algorithm (Many-objective Optimization):
    #
    # 1. Many-objective Challenge:
    #    - With 3+ objectives, most solutions become non-dominated
    #    - Traditional Pareto ranking loses selection pressure
    #    - Crowding distance becomes less effective
    #    - Need structured diversity preservation
    #
    # 2. NSGA-III Innovations:
    #    - Reference points on normalized hyperplane
    #    - Associate solutions with reference points
    #    - Select solutions to maintain balanced distribution
    #    - Adaptive normalization for different objective scales
    #
    # 3. Reference Point Strategy:
    #    - Systematic placement on unit simplex
    #    - Each reference point guides search direction
    #    - Solutions clustered around reference points
    #    - Maintains diversity across objective space
    #
    # 4. Selection Mechanism:
    #    - Non-dominated sorting (like NSGA-II)
    #    - Reference point association
    #    - Niche count balancing
    #    - Preserve solutions near each reference point


def main():
    # === NSGAIIISampler Example ===
    # Many-objective Optimization with NSGA-III

    nsga_iii_theory()

    # Load dataset
    X, y = load_breast_cancer(return_X_y=True)
    print(
        f"Dataset: Breast cancer classification ({X.shape[0]} samples, {X.shape[1]} features)"
    )

    # Create many-objective experiment
    experiment = ManyObjectiveExperiment(X, y)

    # Many-objective Problem (4 objectives):
    #   Objective 1: Maximize classification accuracy
    #   Objective 2: Minimize model complexity (tree depth)
    #   Objective 3: Minimize training time
    #   Objective 4: Maximize interpretability (smaller trees)
    #   → Complex trade-offs between multiple conflicting goals

    # Define search space
    param_space = {
        "max_depth": (1, 20),  # Tree depth
        "min_samples_split": (2, 50),  # Minimum samples to split
        "min_samples_leaf": (1, 20),  # Minimum samples per leaf
        "max_leaf_nodes": (10, 200),  # Maximum leaf nodes
        "criterion": ["gini", "entropy"],  # Split criterion
    }

    # Search Space:
    # for param, space in param_space.items():
    #   print(f"  {param}: {space}")

    # Configure NSGAIIISampler
    optimizer = NSGAIIISampler(
        param_space=param_space,
        n_trials=60,  # More trials needed for many objectives
        random_state=42,
        experiment=experiment,
        population_size=24,  # Larger population for many objectives
        mutation_prob=0.1,  # Mutation probability
        crossover_prob=0.9,  # Crossover probability
    )

    # NSGAIIISampler Configuration:
    # n_trials: configured above
    # population_size: larger for many objectives
    # mutation_prob: mutation probability
    # crossover_prob: crossover probability
    # Selection: Reference point-based diversity preservation

    # Note: NSGA-III is designed for 3+ objectives.
    # For 2 objectives, NSGA-II is typically preferred.
    # This example demonstrates the interface for many-objective problems.

    # Run optimization
    # Running NSGA-III many-objective optimization...

    try:
        best_params = optimizer.run()

        # Results
        print("\n=== Results ===")
        print(f"Best parameters: {best_params}")
        print(f"Best score: {optimizer.best_score_:.4f}")
        print()

        # NSGA-III produces a diverse set of solutions across 4D Pareto front:
        #  High accuracy, complex, slower models
        #  Balanced accuracy/complexity trade-offs
        #  Fast, simple, interpretable models
        #  Various combinations optimizing different objectives

    except Exception as e:
        print(f"Many-objective optimization example: {e}")
        print("Note: This demonstrates the interface for many-objective problems.")
        return None, None

    # NSGA-III vs NSGA-II for Many Objectives:
    #
    # NSGA-II Limitations (3+ objectives):
    #  Most solutions become non-dominated
    #  Crowding distance loses effectiveness
    #  Selection pressure decreases
    #  Uneven distribution in objective space

    # NSGA-III Advantages:
    #  Reference points guide search directions
    #  Maintains diversity across all objectives
    #  Better selection pressure in many objectives
    #  Structured exploration of objective space
    #  Adaptive normalization handles different scales

    # Reference Point Mechanism:
    #  Systematic placement on normalized hyperplane
    #  Each point represents a different objective priority
    #  Solutions associated with nearest reference points
    #  Selection maintains balance across all points
    #  Prevents clustering in limited objective regions

    # Many-objective Problem Characteristics:
    #
    # Challenges:
    #  Exponential growth of non-dominated solutions
    #  Difficulty visualizing high-dimensional trade-offs
    #  User preference articulation becomes complex
    #  Increased computational requirements

    # Best Use Cases:
    #  Engineering design with multiple constraints
    #  Multi-criteria decision making (3+ criteria)
    #  Resource allocation problems
    #  System optimization with conflicting requirements
    #  When objective interactions are complex
    #
    # Algorithm Selection Guide:
    #
    # Use NSGA-III when:
    #    3 or more objectives
    #    Objectives are truly conflicting
    #    Comprehensive trade-off analysis needed
    #    Reference point guidance is beneficial
    #
    # Use NSGA-II when:
    #    2 objectives
    #    Simpler Pareto front structure
    #    Established performance for bi-objective problems
    #
    # Use single-objective methods when:
    #    Can formulate as weighted combination
    #    Clear primary objective with constraints
    #    Computational efficiency is critical

    if "best_params" in locals():
        return best_params, optimizer.best_score_
    else:
        return None, None


if __name__ == "__main__":
    best_params, best_score = main()
