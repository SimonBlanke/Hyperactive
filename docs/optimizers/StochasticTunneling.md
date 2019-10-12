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

<p align="center">
<img src="./plots/search_path_Stochastic Tunneling.png" width="1000"/>
</p>

#### Use case/properties:
- When you have a good initial point to start from, but expect the surrounding search space to be very complex
- Good as a second method of optimization
