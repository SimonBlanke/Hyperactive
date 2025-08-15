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
        complexity = params['n_estimators'] * params.get('max_depth', 10) * self.X.shape[1]
        
        # NSGA-II minimizes objectives, so we return both as minimization
        # Note: This is a simplified multi-objective setup for demonstration
        return [-accuracy, complexity / 10000]  # Scale complexity for better balance


def nsga_ii_theory():
    """Explain NSGA-II algorithm theory."""
    print("NSGA-II Algorithm (Multi-objective Optimization):")
    print()
    print("1. Core Concepts:")
    print("   - Pareto Dominance: Solution A dominates B if A is better in all objectives")
    print("   - Pareto Front: Set of non-dominated solutions")
    print("   - Trade-offs: Improving one objective may worsen another")
    print()
    print("2. NSGA-II Process:")
    print("   - Initialize population randomly")
    print("   - For each generation:")
    print("     a) Fast non-dominated sorting (rank solutions by dominance)")
    print("     b) Crowding distance calculation (preserve diversity)")
    print("     c) Selection based on rank and crowding distance")
    print("     d) Crossover and mutation to create offspring")
    print()
    print("3. Selection Criteria:")
    print("   - Primary: Non-domination rank (prefer better fronts)")
    print("   - Secondary: Crowding distance (prefer diverse solutions)")
    print("   - Elitist: Best solutions always survive")
    print()
    print("4. Output:")
    print("   - Set of Pareto-optimal solutions")
    print("   - User chooses final solution based on preferences")
    print()


def main():
    print("=== NSGAIISampler Example ===")
    print("Multi-objective Optimization with NSGA-II")
    print()
    
    nsga_ii_theory()
    
    # Load dataset
    X, y = load_digits(return_X_y=True)
    print(f"Dataset: Handwritten digits ({X.shape[0]} samples, {X.shape[1]} features)")
    
    # Create multi-objective experiment
    experiment = MultiObjectiveExperiment(X, y)
    
    print("Multi-objective Problem:")
    print("  Objective 1: Maximize classification accuracy")
    print("  Objective 2: Minimize model complexity")
    print("  → Trade-off between performance and simplicity")
    print()
    
    # Define search space
    param_space = {
        "n_estimators": (10, 200),           # Number of trees
        "max_depth": (1, 20),                # Tree depth (complexity)
        "min_samples_split": (2, 20),        # Minimum samples to split
        "min_samples_leaf": (1, 10),         # Minimum samples per leaf
        "max_features": ["sqrt", "log2", None],  # Feature sampling
    }
    
    print("Search Space:")
    for param, space in param_space.items():
        print(f"  {param}: {space}")
    print()
    
    # Configure NSGAIISampler
    optimizer = NSGAIISampler(
        param_space=param_space,
        n_trials=50,  # Population evolves over multiple generations
        random_state=42,
        experiment=experiment,
        population_size=20,  # Population size for genetic algorithm
        mutation_prob=0.1,   # Mutation probability
        crossover_prob=0.9   # Crossover probability
    )
    
    print("NSGAIISampler Configuration:")
    print(f"  n_trials: {optimizer.n_trials}")
    print(f"  population_size: {optimizer.population_size}")
    print(f"  mutation_prob: {optimizer.mutation_prob}")
    print(f"  crossover_prob: {optimizer.crossover_prob}")
    print("  Selection: Non-dominated sorting + crowding distance")
    print()
    
    print("Note: This example demonstrates the interface.")
    print("In practice, NSGA-II returns multiple Pareto-optimal solutions.")
    print("For single-objective problems, consider TPE or GP samplers instead.")
    print()
    
    # Run optimization
    print("Running NSGA-II multi-objective optimization...")
    
    try:
        best_params = optimizer.run()
        
        # Results
        print("\n=== Results ===")
        print(f"Best parameters: {best_params}")
        print(f"Best score: {optimizer.best_score_:.4f}")
        print()
        
        print("NSGA-II typically returns multiple solutions along Pareto front:")
        print("• High accuracy, high complexity models")
        print("• Medium accuracy, medium complexity models")
        print("• Lower accuracy, low complexity models")
        print("• User selects based on preferences/constraints")
        
    except Exception as e:
        print(f"Multi-objective optimization example: {e}")
        print("Note: This demonstrates the interface for multi-objective problems.")
        return None, None
    
    # NSGA-II behavior analysis
    print("\nNSGA-II Evolution Process:")
    print()
    print("Generation 1: Random initialization")
    print("• Diverse population across parameter space")
    print("• Wide range of accuracy/complexity trade-offs")
    print()
    
    print("Generations 2-N: Evolutionary improvement")
    print("• Non-dominated sorting identifies best fronts")
    print("• Crowding distance maintains solution diversity")
    print("• Crossover combines good solutions")
    print("• Mutation explores new parameter regions")
    print()
    
    print("Final Population: Pareto front approximation")
    print("• Multiple non-dominated solutions")
    print("• Represents optimal trade-offs")
    print("• User chooses based on domain requirements")
    print()
    
    print("Key Advantages:")
    print("✓ Handles multiple conflicting objectives naturally")
    print("✓ Finds diverse set of optimal trade-offs")
    print("✓ No need to specify objective weights a priori")
    print("✓ Provides insight into objective relationships")
    print("✓ Robust to objective scaling differences")
    print()
    
    print("Best Use Cases:")
    print("• True multi-objective problems (accuracy vs speed, cost vs quality)")
    print("• When trade-offs between objectives are important")
    print("• Robustness analysis with multiple criteria")
    print("• When single objective formulation is unclear")
    print()
    
    print("Limitations:")
    print("• More complex than single-objective methods")
    print("• Requires more evaluations (population-based)")
    print("• May be overkill for single-objective problems")
    print("• Final solution selection still required")
    print()
    
    print("When to Use NSGA-II vs Single-objective Methods:")
    print("Use NSGA-II when:")
    print("  • Multiple objectives genuinely conflict")
    print("  • Trade-off analysis is valuable")
    print("  • Objective weights are unknown")
    print()
    print("Use TPE/GP when:")
    print("  • Single clear objective")
    print("  • Computational budget is limited")
    print("  • Faster convergence needed")
    
    if 'best_params' in locals():
        return best_params, optimizer.best_score_
    else:
        return None, None


if __name__ == "__main__":
    best_params, best_score = main()