"""Base classes for optimizers and experiments."""

from hyperactive.base import BaseExperiment
from ._utility import add_callback, add_catch

__all__ = ["BaseExperiment", "add_callback", "add_catch"]
