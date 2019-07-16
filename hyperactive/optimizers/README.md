# Optimization Techniques

Hyperactive offers a wide variety of basic, meta-heuristic and sequential model-based optimization techniques for machine learning model selection and hyperparameter tuning. This readme provides an overview[*](https://github.com/SimonBlanke/Hyperactive/tree/master/hyperactive/optimizers#disclaimer) and brief explainations of those techniques and proposes a possible field of application.



---

<p align="center">
  <a href="https://github.com/SimonBlanke/Hyperactive/tree/master/hyperactive/optimizers#local-search">Local Search</a> |
  <a href="https://github.com/SimonBlanke/Hyperactive/tree/master/hyperactive/optimizers#random-methods">Random Methods</a> |
  <a href="https://github.com/SimonBlanke/Hyperactive/tree/master/hyperactive/optimizers#markov-chain-monte-carlo">Markov Chain Monte Carlo</a> |
  <a href="https://github.com/SimonBlanke/Hyperactive/tree/master/hyperactive/optimizers#population-methods">Population Methods</a> |
  <a href="https://github.com/SimonBlanke/Hyperactive/tree/master/hyperactive/optimizers#sequential-methods">Sequential Methods</a>
</p>

---


## Local Search:

#### Hill Climbing
Hill climbing is a very basic optimization technique, that explores the search space only localy. It starts at an initial point, which is often chosen randomly and continues to move to positions with a better solution. It has no method against getting stuck in local optima.

###### When to use:
- Never as a first method of optimization
- When you have a very good initial point to start from
- If the search space is very simple and has few local optima or saddle points

#### Stochastic Hill Climbing
Stochastic hill climbing extends the normal hill climbing by a simple method against getting stuck in local optima. It has a parameter you can set, that determines the probability to accept worse solutions as a next position.

###### When to use:
- Never as a first method of optimization
- When you have a very good initial point to start from

#### Tabu Search
Tabu search is a metaheuristic method, that explores new positions like hill climbing but memorizes previous positions and avoids those. This helps finding new trajectories through the search space.

###### When to use:
- When you have a good initial point to start from

## Random Methods:

#### Random Search
The random search explores by choosing a new position at random after each iteration. Some random seach implementations choose a new position within a large hypersphere around the current position. The implementation in hyperactive is purely random across the search space in each step.

###### When to use:
- Very good as a first method of optimization or to start exploring the search space
- For a short optimization run to get an acceptable solution

#### Random Restart Hill Climbing
Random restart hill climbing works by starting a hill climbing search and jumping to a random new position after a number of iterations.

###### When to use:
- Good as a first method of optimization
- For a short optimization run to get an acceptable solution

#### Random Annealing
An algorithm that chooses a new position within a large hypersphere around the current position. This hypersphere gets smaller over time.

###### When to use:
- Disclaimer: I have not seen this algorithm before, but invented it my self. It seems to be a good alternative to the other random algorithms
- Good as a first method of optimization
- For a short optimization run to get an acceptable solution

## Markov Chain Monte Carlo:

#### Simulated Annealing
Simulated annealing chooses its next possible position similar to hill climbing, but it accepts worse results with a probability that decreases with time:

![equation](https://latex.codecogs.com/gif.latex?exp%5Cleft%20%28%20-%20%5Cfrac%7Bf%28y%29%20-%20f%28x%29%20%7D%7BT%7D%20%5Cright%20%29)  

It simulates a temperature that decreases with each iteration, similar to a material cooling down.

###### When to use:


#### Stochastic Tunneling

###### When to use:


#### Parallel Tempering

###### When to use:


## Population Methods:

#### Particle Swarm Optimizer
Particle swarm optimization works by initializing a number of positions at the same time and moving all of those closer to the best one after each iteration. 

###### When to use:
- If the search space is complex and large
- You have enough time for many model evaluations

#### Evolution Strategy
Evolution strategy mutates and combines the best individuals of a population across a number of generations without transforming them into an array of bits (like genetic algorithms) but uses the real values of the positions. 

###### When to use:
- If the search space is very complex and large
- You have enough time for many model evaluations

## Sequential Methods:

#### Bayesian Optimization
Bayesian optimization chooses new positions by calculating the expected improvement of every position in the search space based on a gaussian process that trains on already evaluated solutions.

###### When to use:
- If model evaluations take a long time
- If you do not want to do many iterations
- If your search space is not to big



##### Disclaimer:
The classification into the categories above is not necessarly scientificly accurate, but aims to provide an idea of the functionality of the methods.

