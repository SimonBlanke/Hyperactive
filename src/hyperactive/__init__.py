# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import importlib.metadata

__version__ = importlib.metadata.version("hyperactive")
__license__ = "MIT"


from .hyperactive import Hyperactive


__all__ = [
    "Hyperactive",
]
