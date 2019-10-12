#### Parallel Tempering

Parallel Tempering initializes multiple simulated annealing seaches with different temperatures and chooses to swap those temperatures with the following probability:

<p align="center">
  <a href="equation">
    <img src="https://latex.codecogs.com/gif.latex?p%20%3D%20%5Cmin%20%5Cleft%20%28%201%2C%20e%5E%7B%5CDelta%20f%20%5Cleft%20%28%20%5Cfrac%7B1%7D%7BT_x%7D%20-%20%5Cfrac%7B1%7D%7BT_y%7D%20%5Cright%20%29%7D%20%5Cright%20%29">
  </a>
</p>


###### Use case/properties:
- Not as dependend of a good initial position as simulated annealing
- If you have enough time for many model evaluations

<p align="center">
<img src="./plots/search_path_Parallel Tempering.png" width="1000"/>
</p>
