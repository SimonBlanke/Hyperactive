## Choose an optimizer

Your decision to use a specific optimizer should be based on the time it takes to evaluate a model and if you already have a start point. Try to stick to the following <b>guideline</b>, when choosing an optimizer:
- only use local or mcmc optimizers, if you have a <b>good start point</b>
- random optimizers are a good way to <b>start exploring</b> the search space
- the majority of the <b>iteration-time</b> should be the <b>evaluation-time</b> of the model

You can choose an optimizer-class from the list provided in the [API](https://github.com/SimonBlanke/Hyperactive#hyperactive-api).
All optimization techniques are explained in more detail [here](https://github.com/SimonBlanke/Hyperactive/tree/master/hyperactive/optimizers#optimization-techniques). A comparison between the iteration- and evaluation-time for different models can be seen [here](https://github.com/SimonBlanke/Hyperactive/tree/master/hyperactive/model#supported-packages).
