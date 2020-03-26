## Hill Climbing

Hill climbing is a very basic optimization technique, that explores the search space only localy. It starts at an initial point, which is often chosen randomly and continues to move to positions with a better solution. It has no method against getting stuck in local optima.

**Available parameters:**
- epsilon
- distribution
- n_neighbours

---

**Use case/properties:**
- Never as a first method of optimization
- When you have a very good initial point to start from
- If the search space is very simple and has few local optima or saddle points

<p align="center">
<img src="./plots/search_paths/HillClimbing [('epsilon', 0.03)].png" width= 49%/>
<img src="./plots/search_paths/HillClimbing [('epsilon', 0.1)].png" width= 49%/>
</p>

<p align="center">
<img src="./plots/search_paths/HillClimbing [('climb_dist', 'laplace')].png" width= 49%/>
<img src="./plots/search_paths/HillClimbing [('climb_dist', 'logistic')].png" width= 49%/>
</p>


<br>

## Stochastic Hill Climbing
Stochastic hill climbing extends the normal hill climbing by a simple method against getting stuck in local optima. It has a parameter you can set, that determines the probability to accept worse solutions as a next position.

**Available parameters:**
- epsilon
- distribution
- n_neighbours
- p_down

---

**Use case/properties:**
- Never as a first method of optimization
- When you have a very good initial point to start from

<p align="center">
<img src="./plots/search_paths/StochasticHillClimbing [('p_down', 0.1)].png" width= 49%/>
<img src="./plots/search_paths/StochasticHillClimbing [('p_down', 0.3)].png" width= 49%/>
</p>

<p align="center">
<img src="./plots/search_paths/StochasticHillClimbing [('p_down', 0.5)].png" width= 49%/>
<img src="./plots/search_paths/StochasticHillClimbing [('p_down', 0.9)].png" width= 49%/>
</p>

<br>

## Tabu Search

Tabu search is a metaheuristic method, that explores new positions like hill climbing but memorizes previous positions and avoids those. This helps finding new trajectories through the search space.

**Available parameters:**
- epsilon
- distribution
- n_neighbours
- tabu_memory

---

**Use case/properties:**
- When you have a good initial point to start from

<p align="center">
<img src="./plots/search_paths/TabuSearch [('tabu_memory', 1)].png" width= 49%/>
<img src="./plots/search_paths/TabuSearch [('tabu_memory', 3)].png" width= 49%/>
</p>

<p align="center">
<img src="./plots/search_paths/TabuSearch [('tabu_memory', 10)].png" width= 49%/>
<img src="./plots/search_paths/TabuSearch [('tabu_memory', 3), ('epsilon', 0.1)].png" width= 49%/>
</p>

<br>

## Simulated Annealing

Simulated annealing chooses its next possible position similar to hill climbing, but it accepts worse results with a probability that decreases with time:

<p align="center">
  <a href="equation">
    <img src="https://latex.codecogs.com/gif.latex?p%20%3D%20exp%20%5Cleft%20%28%20-%5Cfrac%7B%5CDelta%20f_%7Bnorm%7D%7D%7BT%7D%20%5Cright%20%29">
  </a>
</p>

It simulates a temperature that decreases with each iteration, similar to a material cooling down. The following normalization is used to calculate the probability independent of the metric:

<p align="center">
  <a href="equation">
    <img src="https://latex.codecogs.com/gif.latex?%5CDelta%20f_%7Bnorm%7D%20%3D%20%5Cfrac%7Bf%28y%29%20-%20f%28y%29%7D%7Bf%28y%29%20&plus;%20f%28y%29%7D">
  </a>
</p>

**Available parameters:**
- epsilon
- distribution
- n_neighbours
- start_temp
- annealing_rate
- norm_factor

---

**Use case/properties:**
- When you have a good initial point to start from, but expect the surrounding search space to be very complex
- Good as a second method of optimization

<p align="center">
<img src="./plots/search_paths/SimulatedAnnealing [('annealing_rate', 0.8)].png" width= 49%/>
<img src="./plots/search_paths/SimulatedAnnealing [('annealing_rate', 0.9)].png" width= 49%/>
</p>


## Random Search

The random search explores by choosing a new position at random after each iteration. Some random search implementations choose a new position within a large hypersphere around the current position. The implementation in hyperactive is purely random across the search space in each step.

---

**Use case/properties:**
- Very good as a first method of optimization or to start exploring the search space
- For a short optimization run to get an acceptable solution

<p align="center">
<img src="./plots/search_paths/RandomSearch.png" width= 49%/>
</p>

<br>

## Random Restart Hill Climbing

Random restart hill climbing works by starting a hill climbing search and jumping to a random new position after a number of iterations.

**Available parameters:**
- epsilon
- distribution
- n_neighbours
- n_restarts

---

**Use case/properties:**
- Good as a first method of optimization
- For a short optimization run to get an acceptable solution

<p align="center">
<img src="./plots/search_paths/RandomRestartHillClimbing [('n_restarts', 5)].png" width= 49%/>
<img src="./plots/search_paths/RandomRestartHillClimbing [('n_restarts', 10)].png" width= 49%/>
</p>

<br>

## Random Annealing

An algorithm that chooses a new position within a large hypersphere around the current position. This hypersphere gets smaller over time.

**Available parameters:**
- epsilon
- distribution
- n_neighbours
- start_temp
- annealing_rate

---

**Use case/properties:**
- Disclaimer: I have not seen this algorithm before, but invented it myself. It seems to be a good alternative to the other random algorithms
- Good as a first method of optimization
- For a short optimization run to get an acceptable solution

<p align="center">
<img src="./plots/search_paths/RandomAnnealing [('epsilon_mod', 3)].png" width= 49%/>
<img src="./plots/search_paths/RandomAnnealing [('epsilon_mod', 10)].png" width= 49%/>
<img src="./plots/search_paths/RandomAnnealing [('epsilon_mod', 25)].png" width= 49%/>
<img src="./plots/search_paths/RandomAnnealing [('epsilon_mod', 25), ('annealing_rate', 0.9)].png" width= 49%/>
</p>



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


## Bayesian Optimization
Bayesian optimization chooses new positions by calculating the expected improvement of every position in the search space based on a gaussian process that trains on already evaluated positions.

**Available parameters:**
- gpr
- xi
- start_up_evals
- skip_retrain
- max_sample_size
- warm_start_smbo

---

**Use case/properties:**
- If model evaluations take a long time
- If you do not want to do many iterations
- If your search space is not to big

<p align="center">
<img src="./plots/search_paths/Bayesian.png" width= 49%/>
</p>

<br>

## Tree of Parzen Estimators
Tree of Parzen Estimators also chooses new positions by calculating the expected improvement. It does so by calculating the ratio of probability being among the best positions and the worst positions. Those probabilities are determined with a kernel density estimator, that is trained on alrady evaluated positions.

**Available parameters:**
- tree_regressor
- gamma_tpe
- start_up_evals
- skip_retrain
- max_sample_size
- warm_start_smbo

---

**Use case/properties:**
- If model evaluations take a long time
- If you do not want to do many iterations
- If your search space is not to big

## Decision Tree Optimizer
