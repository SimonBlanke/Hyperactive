## Parallel Tempering

Parallel Tempering initializes multiple simulated annealing searches with different temperatures and chooses to swap those temperatures with the following probability:

<p align="center">
  <a href="equation">
    <img src="https://latex.codecogs.com/gif.latex?p%20%3D%20%5Cmin%20%5Cleft%20%28%201%2C%20e%5E%7B%5CDelta%20f%20%5Cleft%20%28%20%5Cfrac%7B1%7D%7BT_x%7D%20-%20%5Cfrac%7B1%7D%7BT_y%7D%20%5Cright%20%29%7D%20%5Cright%20%29">
  </a>
</p>

**Available parameters:**
- epsilon
- distribution
- n_neighbours
- annealing_rate
- system_temperatures
- n_swaps

---

**Use case/properties:**
- Not as dependend of a good initial position as simulated annealing
- If you have enough time for many model evaluations

<p align="center">
<img src="./plots/search_paths/ParallelTempering [('system_temperatures', [0.1, 1, 10, 100])].png" width= 49%/>
<img src="./plots/search_paths/ParallelTempering [('system_temperatures', [0.01, 100])].png" width= 49%/>
</p>

<br>

## Particle Swarm Optimization

Particle swarm optimization works by initializing a number of positions at the same time and moving all of those closer to the best one after each iteration.

**Available parameters:**
- n_particles
- inertia
- cognitive_weight
- social_weight

---

**Use case/properties:**
- If the search space is complex and large
- If you have enough time for many model evaluations

<p align="center">
<img src="./plots/search_paths/ParticleSwarm [('n_particles', 4)].png" width= 49%/>
<img src="./plots/search_paths/ParticleSwarm [('n_particles', 10)].png" width= 49%/>
</p>

<br>

## Evolution Strategy
Evolution strategy mutates and combines the best individuals of a population across a number of generations without transforming them into an array of bits (like genetic algorithms) but uses the real values of the positions.

**Available parameters:**
- individuals
- mutation_rate
- crossover_rate

---

**Use case/properties:**
- If the search space is very complex and large
- If you have enough time for many model evaluations

<p align="center">
<img src="./plots/search_paths/EvolutionStrategy [('individuals', 4)].png" width= 49%/>
<img src="./plots/search_paths/EvolutionStrategy [('individuals', 10)].png" width= 49%/>
<img src="./plots/search_paths/EvolutionStrategy [('individuals', 10), ('mutation_rate', 0.1), ('crossover_rate', 0.9)].png" width= 49%/>
<img src="./plots/search_paths/EvolutionStrategy [('individuals', 10), ('mutation_rate', 0.9), ('crossover_rate', 0.1)].png" width= 49%/>
</p>
