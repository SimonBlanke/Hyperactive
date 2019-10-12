#### Simulated Annealing

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

###### Use case/properties:
- When you have a good initial point to start from, but expect the surrounding search space to be very complex
- Good as a second method of optimization

<p align="center">
<img src="./plots/search_path_Simulated Annealing.png" width="1000"/>
</p>
