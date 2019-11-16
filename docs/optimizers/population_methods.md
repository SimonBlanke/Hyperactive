## Particle Swarm Optimization

Particle swarm optimization works by initializing a number of positions at the same time and moving all of those closer to the best one after each iteration.

---

**Use case/properties:**
- If the search space is complex and large
- If you have enough time for many model evaluations

<p align="center">
<img src="./plots/search_path_ParticleSwarm.svg" width="1200"/>
</p>


## Evolution Strategy
Evolution strategy mutates and combines the best individuals of a population across a number of generations without transforming them into an array of bits (like genetic algorithms) but uses the real values of the positions.

---

**Use case/properties:**
- If the search space is very complex and large
- If you have enough time for many model evaluations

<p align="center">
<img src="./plots/search_path_EvolutionStrategy.svg" width="1200"/>
</p>
