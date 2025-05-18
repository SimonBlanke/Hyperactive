"""Individual optimization algorithms."""
# copyright: hyperactive developers, MIT License (see LICENSE file)

from hyperactive.opt.gfo import HillClimbing
from hyperactive.opt.gridsearch import GridSearch

__all__ = [
    "GridSearch",
    "HillClimbing",
]
