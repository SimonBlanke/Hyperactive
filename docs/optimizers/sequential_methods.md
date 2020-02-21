## Bayesian Optimization
Bayesian optimization chooses new positions by calculating the expected improvement of every position in the search space based on a gaussian process that trains on already evaluated positions.

---

**Use case/properties:**
- If model evaluations take a long time
- If you do not want to do many iterations
- If your search space is not to big

<p align="center">
<img src="./plots/search_paths/Bayesian.png" width= 49%/>
</p>

<br>


## Tree of Parzen Estimators
Tree of Parzen Estimators also chooses new positions by calculating the expected improvement. It does so by calculating the ratio of probability being among the best positions and the worst positions. Those probabilities are determined with a kernel density estimator, that is trained on alrady evaluated positions.

---

**Use case/properties:**
- If model evaluations take a long time
- If you do not want to do many iterations
- If your search space is not to big
