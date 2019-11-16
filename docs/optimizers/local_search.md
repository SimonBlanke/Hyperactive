## Hill Climbing

Hill climbing is a very basic optimization technique, that explores the search space only localy. It starts at an initial point, which is often chosen randomly and continues to move to positions with a better solution. It has no method against getting stuck in local optima.

---

**Use case/properties:**
- Never as a first method of optimization
- When you have a very good initial point to start from
- If the search space is very simple and has few local optima or saddle points

<p align="center">
<img src="./plots/search_path_HillClimbing.svg" width="680"/>
<img src="./plots/search_path_HillClimbing({'epsilon': 0.1}).svg" width="680"/>
</p>


## Stochastic Hill Climbing
Stochastic hill climbing extends the normal hill climbing by a simple method against getting stuck in local optima. It has a parameter you can set, that determines the probability to accept worse solutions as a next position.

---

**Use case/properties:**
- Never as a first method of optimization
- When you have a very good initial point to start from

<p align="center">
<img src="./plots/search_path_StochasticHillClimbing.svg" width="800"/>
</p>


## Tabu Search

Tabu search is a metaheuristic method, that explores new positions like hill climbing but memorizes previous positions and avoids those. This helps finding new trajectories through the search space.

---

**Use case/properties:**
- When you have a good initial point to start from

<p align="center">
<img src="./plots/search_path_TabuSearch.svg" width="800"/>
</p>
