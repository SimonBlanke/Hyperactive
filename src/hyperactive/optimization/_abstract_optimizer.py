from abc import ABC, abstractmethod
from typing import Union, List, Dict, Type
from skbase.base import BaseObject


class AbstractOptimizer(ABC, BaseObject):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def add_search(self, *args, **kwargs):
        pass

    @abstractmethod
    def run(self, *args, **kwargs):
        pass
