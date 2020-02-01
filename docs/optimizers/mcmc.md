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

---

**Use case/properties:**
- When you have a good initial point to start from, but expect the surrounding search space to be very complex
- Good as a second method of optimization

<p align="center">
<img src="./plots/search_paths/SimulatedAnnealing [('annealing_rate', 0.8)].png" width= 49%/>
<img src="./plots/search_paths/SimulatedAnnealing [('annealing_rate', 0.9)].png" width= 49%/>
</p>


## Stochastic Tunneling

Stochastic Tunneling works very similar to simulated annealing, but modifies its probability to accept worse solutions by an additional term:

<p align="center">
  <a href="equation">
    <img src="https://latex.codecogs.com/gif.latex?f_%7BSTUN%7D%20%3D%201%20-%20exp%28-%5Cgamma%20%5CDelta%20f%29">
  </a>
</p>

This new acceptance factor is used instead of the delta f in the original equation:

<p align="center">
  <a href="equation">
    <img src="https://latex.codecogs.com/gif.latex?p%20%3D%20exp%28-%5Cbeta%20f_%7BSTUN%7D%20%29">
  </a>
</p>

---

**Use case/properties:**
- When you have a good initial point to start from, but expect the surrounding search space to be very complex
- Good as a second method of optimization

<p align="center">
<img src="./plots/search_paths/StochasticTunneling [('gamma', 0.1)].png" width= 49%/>
<img src="./plots/search_paths/StochasticTunneling [('gamma', 3)].png" width= 49%/>
</p>


## Parallel Tempering

Parallel Tempering initializes multiple simulated annealing searches with different temperatures and chooses to swap those temperatures with the following probability:

<p align="center">
  <a href="equation">
    <img src="https://latex.codecogs.com/gif.latex?p%20%3D%20%5Cmin%20%5Cleft%20%28%201%2C%20e%5E%7B%5CDelta%20f%20%5Cleft%20%28%20%5Cfrac%7B1%7D%7BT_x%7D%20-%20%5Cfrac%7B1%7D%7BT_y%7D%20%5Cright%20%29%7D%20%5Cright%20%29">
  </a>
</p>

---

**Use case/properties:**
- Not as dependend of a good initial position as simulated annealing
- If you have enough time for many model evaluations

<p align="center">
<img src="./plots/search_paths/ParallelTempering [('system_temperatures', [0.1, 1, 10, 100])].png" width= 49%/>
<img src="./plots/search_paths/ParallelTempering [('system_temperatures', [0.01, 100])].png" width= 49%/>
</p>
