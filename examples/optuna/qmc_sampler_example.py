"""
QMCSampler Example - Quasi-Monte Carlo Sampling

The QMCSampler uses Quasi-Monte Carlo sequences (like Sobol or Halton) 
to generate low-discrepancy samples. These sequences provide better 
coverage of the parameter space compared to purely random sampling.

Characteristics:
- Low-discrepancy sequences for uniform space filling
- Better convergence than random sampling
- Deterministic sequence generation
- Excellent space coverage properties
- No learning from previous evaluations
- Good baseline for comparison with adaptive methods

QMC sequences are particularly effective for:
- Integration and sampling problems
- Initial design of experiments
- Baseline optimization comparisons
"""

import numpy as np
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from hyperactive.experiment.integrations import SklearnCvExperiment
from hyperactive.opt.optuna import QMCSampler


def qmc_theory():
    """Explain Quasi-Monte Carlo theory."""
    # Quasi-Monte Carlo (QMC) Theory:
    #
    # 1. Low-Discrepancy Sequences:
    #    - Deterministic sequences that fill space uniformly
    #    - Better distribution than random sampling
    #    - Minimize gaps and clusters in parameter space
    #    - Convergence rate O(log^d(N)/N) vs O(1/√N) for random
    #
    # 2. Common QMC Sequences:
    #    - Sobol: Based on binary representations
    #    - Halton: Based on prime number bases
    #    - Faure: Generalization of Halton sequences
    #    - Each has different strengths for different dimensions
    #
    # 3. Space-Filling Properties:
    #    - Stratification: Even coverage of parameter regions
    #    - Low discrepancy: Uniform distribution approximation
    #    - Correlation breaking: Reduces clustering
    #
    # 4. Advantages over Random Sampling:
    #    - Better convergence for integration
    #    - More uniform exploration
    #    - Reproducible sequences
    #    - No unlucky clustering


def demonstrate_space_filling():
    """Demonstrate space-filling properties conceptually."""
    print("Space-Filling Comparison:")
    print()
    print("Random Sampling:")
    print("  • Can have clusters and gaps")
    print("  • Uneven coverage especially with few samples")
    print("  • Variance in coverage quality")
    print("  • Some regions may be under-explored")
    print()
    print("Quasi-Monte Carlo (QMC):")
    print("  • Systematic space filling")
    print("  • Even coverage with fewer samples")
    print("  • Consistent coverage quality")
    print("  • All regions explored proportionally")
    print()
    print("Grid Sampling:")
    print("  • Perfect regular coverage")
    print("  • Exponential scaling with dimensions")
    print("  • May miss optimal points between grid lines")
    print()
    print("→ QMC provides balanced approach between random and grid")
    print()


def main():
    print("=== QMCSampler Example ===")
    print("Quasi-Monte Carlo Low-Discrepancy Sampling")
    print()
    
    qmc_theory()
    demonstrate_space_filling()
    
    # Load dataset
    X, y = load_wine(return_X_y=True)
    print(f"Dataset: Wine classification ({X.shape[0]} samples, {X.shape[1]} features)")
    
    # Create experiment
    estimator = LogisticRegression(random_state=42, max_iter=1000)
    experiment = SklearnCvExperiment(estimator=estimator, X=X, y=y, cv=5)
    
    # Define search space
    param_space = {
        "C": (0.001, 100),              # Regularization strength
        "l1_ratio": (0.0, 1.0),         # Elastic net mixing parameter
        "solver": ["liblinear", "saga"], # Solver algorithm
        "penalty": ["l1", "l2", "elasticnet"],  # Regularization type
    }
    
    print("Search Space:")
    for param, space in param_space.items():
        print(f"  {param}: {space}")
    print()
    
    # Configure QMCSampler
    optimizer = QMCSampler(
        param_space=param_space,
        n_trials=32,  # Power of 2 often works well for QMC
        random_state=42,
        experiment=experiment,
        qmc_type="sobol",    # Sobol or Halton sequences
        scramble=True        # Randomized QMC (Owen scrambling)
    )
    
    print("QMCSampler Configuration:")
    print(f"  n_trials: {optimizer.n_trials} (power of 2 for better QMC properties)")
    print(f"  qmc_type: '{optimizer.qmc_type}' sequence")
    print(f"  scramble: {optimizer.scramble} (randomized QMC)")
    print("  Deterministic low-discrepancy sampling")
    print()
    
    print("QMC Sequence Types:")
    print("• Sobol: Excellent for moderate dimensions, binary-based")
    print("• Halton: Good for low dimensions, prime-based")
    print("• Scrambling: Adds randomization while preserving uniformity")
    print()
    
    # Run optimization
    print("Running QMC sampling optimization...")
    best_params = optimizer.run()
    
    # Results
    print("\n=== Results ===")
    print(f"Best parameters: {best_params}")
    print(f"Best score: {optimizer.best_score_:.4f}")
    print()
    
    # QMC behavior analysis
    print("QMC Sampling Analysis:")
    print()
    print("Sequence Properties:")
    print("✓ Deterministic generation (reproducible with same seed)")
    print("✓ Low-discrepancy (uniform distribution approximation)")
    print("✓ Space-filling (systematic coverage of parameter space)")
    print("✓ Stratification (even coverage of all regions)")
    print("✓ No clustering or large gaps")
    print()
    
    print("Sobol Sequence Characteristics:")
    print("• Binary-based construction")
    print("• Good equidistribution properties")
    print("• Effective for dimensions up to ~40")
    print("• Popular choice for QMC sampling")
    print()
    
    if optimizer.scramble:
        print("Scrambling Benefits:")
        print("• Breaks regularity patterns")
        print("• Provides Monte Carlo error estimates")
        print("• Maintains low-discrepancy properties")
        print("• Reduces correlation artifacts")
        print()
    
    print("QMC vs Other Sampling Methods:")
    print()
    print("vs Pure Random Sampling:")
    print("  + Better space coverage with fewer samples")
    print("  + More consistent performance")
    print("  + Faster convergence for integration-like problems")
    print("  - No true randomness (if needed for some applications)")
    print()
    print("vs Grid Search:")
    print("  + Works well in higher dimensions")
    print("  + No exponential scaling")
    print("  + Covers continuous spaces naturally")
    print("  - No systematic guarantee of finding grid optimum")
    print()
    print("vs Adaptive Methods (TPE, GP):")
    print("  + No assumptions about objective function")
    print("  + Embarrassingly parallel")
    print("  + Consistent performance regardless of function type")
    print("  - No learning from previous evaluations")
    print("  - May waste evaluations in clearly suboptimal regions")
    print()
    
    print("Best Use Cases:")
    print("• Design of experiments (DoE)")
    print("• Initial exploration phase")
    print("• Baseline for comparing adaptive methods")
    print("• Integration and sampling problems")
    print("• When function evaluations are parallelizable")
    print("• Robustness testing across parameter space")
    print()
    
    print("Implementation Considerations:")
    print("• Use powers of 2 for n_trials with Sobol sequences")
    print("• Consider scrambling for better statistical properties")
    print("• Choose sequence type based on dimensionality:")
    print("  - Sobol: Good general choice")
    print("  - Halton: Better for low dimensions (< 6)")
    print("• QMC works best with transformed uniform parameters")
    print()
    
    print("Practical Recommendations:")
    print("1. Use QMC for initial exploration (first 20-50 evaluations)")
    print("2. Switch to adaptive methods (TPE/GP) for focused search")
    print("3. Use for sensitivity analysis across full parameter space")
    print("4. Good choice when unsure about objective function properties")
    print("5. Ideal for parallel evaluation scenarios")
    
    return best_params, optimizer.best_score_


if __name__ == "__main__":
    best_params, best_score = main()