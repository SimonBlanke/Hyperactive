#### Bayesian Optimization
Bayesian optimization chooses new positions by calculating the expected improvement of every position in the search space based on a gaussian process that trains on already evaluated solutions.

###### Use case/properties:
- If model evaluations take a long time
- If you do not want to do many iterations
- If your search space is not to big

<p align="center">
<img src="./plots/search_path_Bayesian Optimization.png" width="1000"/>
</p>

<br>
