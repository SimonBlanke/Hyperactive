## Random Search

The random search explores by choosing a new position at random after each iteration. Some random search implementations choose a new position within a large hypersphere around the current position. The implementation in hyperactive is purely random across the search space in each step.

---

**Use case/properties:**
- Very good as a first method of optimization or to start exploring the search space
- For a short optimization run to get an acceptable solution

<p align="center">
<img src="./plots/search_path_RandomSearch.svg" width="800"/>
</p>


## Random Restart Hill Climbing

Random restart hill climbing works by starting a hill climbing search and jumping to a random new position after a number of iterations.

---

**Use case/properties:**
- Good as a first method of optimization
- For a short optimization run to get an acceptable solution

<p align="center">
<img src="./plots/search_path_RandomRestartHillClimbing.svg" width="800"/>
</p>


## Random Annealing

An algorithm that chooses a new position within a large hypersphere around the current position. This hypersphere gets smaller over time.

---

**Use case/properties:**
- Disclaimer: I have not seen this algorithm before, but invented it myself. It seems to be a good alternative to the other random algorithms
- Good as a first method of optimization
- For a short optimization run to get an acceptable solution

<p align="center">
<img src="./plots/search_path_RandomAnnealing.svg" width="800"/>
</p>
