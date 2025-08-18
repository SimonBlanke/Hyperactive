"""
QMCOptimizer Example - Quasi-Monte Carlo Sampling

The QMCOptimizer uses Quasi-Monte Carlo sequences (like Sobol or Halton)
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
from hyperactive.opt.optuna import QMCOptimizer


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
    # Space-Filling Comparison:
    #
    # Random Sampling:
    #    Can have clusters and gaps
    #    Uneven coverage especially with few samples
    #    Variance in coverage quality
    #    Some regions may be under-explored
    #
    # Quasi-Monte Carlo (QMC):
    #    Systematic space filling
    #    Even coverage with fewer samples
    #    Consistent coverage quality
    #    All regions explored proportionally
    #
    # Grid Sampling:
    #    Perfect regular coverage
    #    Exponential scaling with dimensions
    #    May miss optimal points between grid lines
    #
    # → QMC provides balanced approach between random and grid


def main():
    # === QMCOptimizer Example ===
    # Quasi-Monte Carlo Low-Discrepancy Sampling

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
        "C": (0.001, 100),  # Regularization strength
        "l1_ratio": (0.0, 1.0),  # Elastic net mixing parameter
        "solver": ["liblinear", "saga"],  # Solver algorithm
        "penalty": ["l1", "l2", "elasticnet"],  # Regularization type
    }

    # Search Space:
    # C: (0.001, 100) - Regularization strength
    # l1_ratio: (0.0, 1.0) - Elastic net mixing parameter
    # solver: ['liblinear', 'saga'] - Solver algorithm
    # penalty: ['l1', 'l2', 'elasticnet'] - Regularization type

    # Configure QMCOptimizer
    optimizer = QMCOptimizer(
        param_space=param_space,
        n_trials=8,  # Power of 2 often works well for QMC
        random_state=42,
        experiment=experiment,
        qmc_type="sobol",  # Sobol or Halton sequences
        scramble=True,  # Randomized QMC (Owen scrambling)
    )

    # QMCOptimizer Configuration:
    # n_trials: 32 (power of 2 for better QMC properties)
    # qmc_type: 'sobol' sequence
    # scramble: True (randomized QMC)
    # Deterministic low-discrepancy sampling

    # QMC Sequence Types:
    #  Sobol: Excellent for moderate dimensions, binary-based
    #  Halton: Good for low dimensions, prime-based
    #  Scrambling: Adds randomization while preserving uniformity

    # Run optimization
    # Running QMC sampling optimization...
    best_params = optimizer.solve()

    # Results
    print("\n=== Results ===")
    print(f"Best parameters: {best_params}")
    print(f"Best score: {optimizer.best_score_:.4f}")
    print()

    # QMC behavior analysis:
    #
    # QMC Sampling Analysis:
    #
    # Sequence Properties:
    #  Deterministic generation (reproducible with same seed)
    #  Low-discrepancy (uniform distribution approximation)
    #  Space-filling (systematic coverage of parameter space)
    #  Stratification (even coverage of all regions)
    #  No clustering or large gaps
    #
    # Sobol Sequence Characteristics:
    #  Binary-based construction
    #  Good equidistribution properties
    #  Effective for dimensions up to ~40
    #  Popular choice for QMC sampling
    #
    # Scrambling Benefits (when enabled):
    #  Breaks regularity patterns
    #  Provides Monte Carlo error estimates
    #  Maintains low-discrepancy properties
    #  Reduces correlation artifacts
    #
    # QMC vs Other Sampling Methods:
    #
    # vs Pure Random Sampling:
    #   + Better space coverage with fewer samples
    #   + More consistent performance
    #   + Faster convergence for integration-like problems
    #   - No true randomness (if needed for some applications)
    #
    # vs Grid Search:
    #   + Works well in higher dimensions
    #   + No exponential scaling
    #   + Covers continuous spaces naturally
    #   - No systematic guarantee of finding grid optimum
    #
    # vs Adaptive Methods (TPE, GP):
    #   + No assumptions about objective function
    #   + Embarrassingly parallel
    #   + Consistent performance regardless of function type
    #   - No learning from previous evaluations
    #   - May waste evaluations in clearly suboptimal regions
    #
    # Best Use Cases:
    #  Design of experiments (DoE)
    #  Initial exploration phase
    #  Baseline for comparing adaptive methods
    #  Integration and sampling problems
    #  When function evaluations are parallelizable
    #  Robustness testing across parameter space
    #
    # Implementation Considerations:
    #  Use powers of 2 for n_trials with Sobol sequences
    #  Consider scrambling for better statistical properties
    #  Choose sequence type based on dimensionality:
    #   - Sobol: Good general choice
    #   - Halton: Better for low dimensions (< 6)
    #  QMC works best with transformed uniform parameters
    #
    # Practical Recommendations:
    # 1. Use QMC for initial exploration (first 20-50 evaluations)
    # 2. Switch to adaptive methods (TPE/GP) for focused search
    # 3. Use for sensitivity analysis across full parameter space
    # 4. Good choice when unsure about objective function properties
    # 5. Ideal for parallel evaluation scenarios

    return best_params, optimizer.best_score_


if __name__ == "__main__":
    best_params, best_score = main()
