"""Toy experiments."""

from hyperactive.experiment.toy._ackley import Ackley
from hyperactive.experiment.toy._branin_hoo import BraninHoo
from hyperactive.experiment.toy._hartmann import Hartmann
from hyperactive.experiment.toy._parabola import Parabola
from hyperactive.experiment.toy._sphere import Sphere

__all__ = [
    "Ackley",
    "BraninHoo",
    "Hartmann",
    "Parabola",
    "Sphere",
]
