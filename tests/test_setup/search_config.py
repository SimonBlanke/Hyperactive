import numpy as np
from hyperactive.search_config import SearchConfig


search_config = SearchConfig(
    x0=list(np.arange(2, 15, 1)),
    x1=list(np.arange(2, 25, 2)),
)
