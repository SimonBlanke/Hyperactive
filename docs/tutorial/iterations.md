## How many iterations?

The number of iterations should be low for your first optimization to get to know the iteration-time.
For the <b>iteration-time</b> you should take the following effects into account:
- A <b>k-fold-crossvalidation</b> increases evaluation-time like training on k-1 times on the training data
- If you lower <b>cv below 1</b> the evaluation will deal with it like a training/validation-split, where cv marks the training data fraction. Therefore lower cv means faster evaluation.
- Some optimizers will do (and need) <b>multiple evaluations</b> per iteration:
  - Particle-swarm-optimization
  - Evoluion strategy
  - Parallel Tempering
- The <b>complexity</b> of the machine-/deep-learning models will heavily influence the evaluation- and therefore iteration-time.
- The <b>number of epochs</b> should probably be kept low. You just want to compare different types of models. Retrain the best model afterwards with more epochs.
