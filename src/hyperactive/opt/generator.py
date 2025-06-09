import os
from pathlib import Path

# List of algorithm names and corresponding class names
algo_info = [
    ("downhill_simplex", "DownhillSimplexOptimizer"),
    ("simulated_annealing", "SimulatedAnnealingOptimizer"),
    ("direct_algorithm", "DirectAlgorithm"),
    ("lipschitz_optimization", "LipschitzOptimizer"),
    ("pattern_search", "PatternSearch"),
    ("random_restart_hill_climbing", "RandomRestartHillClimbingOptimizer"),
    ("random_search", "RandomSearchOptimizer"),
    ("powells_method", "PowellsMethod"),
    ("differential_evolution", "DifferentialEvolutionOptimizer"),
    ("evolution_strategy", "EvolutionStrategyOptimizer"),
    ("genetic_algorithm", "GeneticAlgorithmOptimizer"),
    ("parallel_tempering", "ParallelTemperingOptimizer"),
    ("particle_swarm_optimization", "ParticleSwarmOptimizer"),
    ("spiral_optimization", "SpiralOptimization"),
    ("bayesian_optimization", "BayesianOptimizer"),
    ("forest_optimizer", "ForestOptimizer"),
    ("tree_structured_parzen_estimators", "TreeStructuredParzenEstimators"),
]

BASE_DIR = Path("generated_opt_algos")


# Template for the Python class file
def create_class_file_content(class_name: str) -> str:
    return f'''from hyperactive.opt._adapters._gfo import _BaseGFOadapter


class {class_name}(_BaseGFOadapter):

    def _get_gfo_class(self):
        """Get the GFO class to use.

        Returns
        -------
        class
            The GFO class to use. One of the concrete GFO classes
        """
        from gradient_free_optimizers import {class_name}

        return {class_name}
'''


# Main generation loop
for name, class_name in algo_info:
    algo_folder = BASE_DIR / name
    algo_folder.mkdir(parents=True, exist_ok=True)

    init_file = algo_folder / "__init__.py"
    class_file = algo_folder / f"_{name}.py"

    # Create __init__.py (empty)
    init_file.touch(exist_ok=True)

    # Write the optimizer class file
    class_file.write_text(create_class_file_content(class_name))

print(f"Generated {len(algo_info)} folders in {BASE_DIR.resolve()}")
