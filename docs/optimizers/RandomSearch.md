## Random Search

The random search explores by choosing a new position at random after each iteration. Some random search implementations choose a new position within a large hypersphere around the current position. The implementation in hyperactive is purely random across the search space in each step.

#### Use case/properties:
- Very good as a first method of optimization or to start exploring the search space
- For a short optimization run to get an acceptable solution

<p align="center">
<img src="./plots/search_path_Random Search.png" width="1000"/>
</p>
