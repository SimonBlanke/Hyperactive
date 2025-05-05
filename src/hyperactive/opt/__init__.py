"""Individual optimization algorithms."""

from hyperactive.gfo import HillClimbing
from hyperactive.opt.gridsearch import GridSearch

__all__ = [
    "GridSearch",
    "HillClimbing",
]
