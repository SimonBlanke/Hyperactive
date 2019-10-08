#### Rosenbrock function

```python
import numpy as np
from hyperactive import Hyperactive

def rosen(para, X, y):
    """The Rosenbrock function"""
    x = np.array([para["x1"], para["x2"], para["x3"], para["x4"]])
    y = np.array([para["x0"], para["x1"], para["x2"], para["x3"]])

    return -sum(100.0*(x-y**2.0)**2.0 + (1-y)**2.0)

x_range = np.arange(0, 3, 0.1)

search_config = {
    rosen: {
        "x0": x_range,
        "x1": x_range,
        "x2": x_range,
        "x3": x_range,
        "x4": x_range,
    }
}

opt = Hyperactive(search_config, n_iter=1000000)
opt.search(0, 0)
```
