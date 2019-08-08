# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from .candidate_sklearn import ScikitLearnCandidate
from .candidate_xgboost import XGBoostCandidate
from .candidate_lightgbm import LightGbmCandidate
from .candidate_catboost import CatBoostCandidate
from .candidate_keras import KerasCandidate
from .candidate_pytorch import PytorchCandidate


__all__ = [
    "ScikitLearnCandidate",
    "XGBoostCandidate",
    "LightGbmCandidate",
    "CatBoostCandidate",
    "KerasCandidate",
    "PytorchCandidate",
]
