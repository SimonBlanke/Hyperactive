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
        complexity = params.get('max_depth', 20)
        
        # Objective 3: Minimize training time
        time_objective = training_time
        
        # Objective 4: Maximize interpretability (minimize tree size)
        # Approximate tree size based on parameters
        max_leaf_nodes = params.get('max_leaf_nodes', 100)
        interpretability = max_leaf_nodes / 100.0  # Normalized
        
        # Return all objectives for minimization (negative accuracy for maximization)
        return [
            -accuracy,           # Minimize negative accuracy (maximize accuracy)
            complexity / 20.0,   # Minimize complexity (normalized)
            time_objective,      # Minimize training time
            interpretability     # Minimize tree size (maximize interpretability)
        ]


def nsga_iii_theory():
    """Explain NSGA-III algorithm theory."""
    print("NSGA-III Algorithm (Many-objective Optimization):")
    print()
    print("1. Many-objective Challenge:")
    print("   - With 3+ objectives, most solutions become non-dominated")
    print("   - Traditional Pareto ranking loses selection pressure")
    print("   - Crowding distance becomes less effective")
    print("   - Need structured diversity preservation")
    print()
    print("2. NSGA-III Innovations:")
    print("   - Reference points on normalized hyperplane")
    print("   - Associate solutions with reference points")
    print("   - Select solutions to maintain balanced distribution")
    print("   - Adaptive normalization for different objective scales")
    print()
    print("3. Reference Point Strategy:")
    print("   - Systematic placement on unit simplex")
    print("   - Each reference point guides search direction")
    print("   - Solutions clustered around reference points")
    print("   - Maintains diversity across objective space")
    print()
    print("4. Selection Mechanism:")
    print("   - Non-dominated sorting (like NSGA-II)")
    print("   - Reference point association")
    print("   - Niche count balancing")
    print("   - Preserve solutions near each reference point")
    print()


def main():
    print("=== NSGAIIISampler Example ===")
    print("Many-objective Optimization with NSGA-III")
    print()
    
    nsga_iii_theory()
    
    # Load dataset
    X, y = load_breast_cancer(return_X_y=True)
    print(f"Dataset: Breast cancer classification ({X.shape[0]} samples, {X.shape[1]} features)")
    
    # Create many-objective experiment
    experiment = ManyObjectiveExperiment(X, y)
    
    print("Many-objective Problem (4 objectives):")
    print("  Objective 1: Maximize classification accuracy")
    print("  Objective 2: Minimize model complexity (tree depth)")
    print("  Objective 3: Minimize training time")
    print("  Objective 4: Maximize interpretability (smaller trees)")
    print("  → Complex trade-offs between multiple conflicting goals")
    print()
    
    # Define search space
    param_space = {
        "max_depth": (1, 20),                # Tree depth
        "min_samples_split": (2, 50),        # Minimum samples to split
        "min_samples_leaf": (1, 20),         # Minimum samples per leaf
        "max_leaf_nodes": (10, 200),         # Maximum leaf nodes
        "criterion": ["gini", "entropy"],    # Split criterion
    }
    
    print("Search Space:")
    for param, space in param_space.items():
        print(f"  {param}: {space}")
    print()
    
    # Configure NSGAIIISampler
    optimizer = NSGAIIISampler(
        param_space=param_space,
        n_trials=60,  # More trials needed for many objectives
        random_state=42,
        experiment=experiment,
        population_size=24,  # Larger population for many objectives
        mutation_prob=0.1,   # Mutation probability
        crossover_prob=0.9   # Crossover probability
    )
    
    print("NSGAIIISampler Configuration:")
    print(f"  n_trials: {optimizer.n_trials}")
    print(f"  population_size: {optimizer.population_size}")
    print(f"  mutation_prob: {optimizer.mutation_prob}")
    print(f"  crossover_prob: {optimizer.crossover_prob}")
    print("  Selection: Reference point-based diversity preservation")
    print()
    
    print("Note: NSGA-III is designed for 3+ objectives.")
    print("For 2 objectives, NSGA-II is typically preferred.")
    print("This example demonstrates the interface for many-objective problems.")
    print()
    
    # Run optimization
    print("Running NSGA-III many-objective optimization...")
    
    try:
        best_params = optimizer.run()
        
        # Results
        print("\n=== Results ===")
        print(f"Best parameters: {best_params}")
        print(f"Best score: {optimizer.best_score_:.4f}")
        print()
        
        print("NSGA-III produces a diverse set of solutions across 4D Pareto front:")
        print("• High accuracy, complex, slower models")
        print("• Balanced accuracy/complexity trade-offs")
        print("• Fast, simple, interpretable models")
        print("• Various combinations optimizing different objectives")
        
    except Exception as e:
        print(f"Many-objective optimization example: {e}")
        print("Note: This demonstrates the interface for many-objective problems.")
        return None, None
    
    # NSGA-III behavior analysis
    print("\nNSGA-III vs NSGA-II for Many Objectives:")
    print()
    print("NSGA-II Limitations (3+ objectives):")
    print("• Most solutions become non-dominated")
    print("• Crowding distance loses effectiveness")
    print("• Selection pressure decreases")
    print("• Uneven distribution in objective space")
    print()
    
    print("NSGA-III Advantages:")
    print("✓ Reference points guide search directions")
    print("✓ Maintains diversity across all objectives")
    print("✓ Better selection pressure in many objectives")
    print("✓ Structured exploration of objective space")
    print("✓ Adaptive normalization handles different scales")
    print()
    
    print("Reference Point Mechanism:")
    print("• Systematic placement on normalized hyperplane")
    print("• Each point represents a different objective priority")
    print("• Solutions associated with nearest reference points")
    print("• Selection maintains balance across all points")
    print("• Prevents clustering in limited objective regions")
    print()
    
    print("Many-objective Problem Characteristics:")
    print()
    print("Challenges:")
    print("• Exponential growth of non-dominated solutions")
    print("• Difficulty visualizing high-dimensional trade-offs")
    print("• User preference articulation becomes complex")
    print("• Increased computational requirements")
    print()
    
    print("Best Use Cases:")
    print("• Engineering design with multiple constraints")
    print("• Multi-criteria decision making (3+ criteria)")
    print("• Resource allocation problems")
    print("• System optimization with conflicting requirements")
    print("• When objective interactions are complex")
    print()
    
    print("Algorithm Selection Guide:")
    print()
    print("Use NSGA-III when:")
    print("  • 3 or more objectives")
    print("  • Objectives are truly conflicting")
    print("  • Comprehensive trade-off analysis needed")
    print("  • Reference point guidance is beneficial")
    print()
    print("Use NSGA-II when:")
    print("  • 2 objectives")
    print("  • Simpler Pareto front structure")
    print("  • Established performance for bi-objective problems")
    print()
    print("Use single-objective methods when:")
    print("  • Can formulate as weighted combination")
    print("  • Clear primary objective with constraints")
    print("  • Computational efficiency is critical")
    
    if 'best_params' in locals():
        return best_params, optimizer.best_score_
    else:
        return None, None


if __name__ == "__main__":
    best_params, best_score = main()