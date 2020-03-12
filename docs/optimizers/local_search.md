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
<img src="./plots/search_paths/HillClimbing [('distribution', 'laplace')].png" width= 49%/>
<img src="./plots/search_paths/HillClimbing [('distribution', 'logistic')].png" width= 49%/>
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
