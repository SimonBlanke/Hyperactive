from .info import InfoLVL0, InfoLVL1
from .progress_bar import ProgressBar, ProgressBarLVL0, ProgressBarLVL1


def set_verbosity(verbosity):
    if verbosity == 0:
        return InfoLVL0, ProgressBar
    elif verbosity == 1:
        return InfoLVL1, ProgressBar
    elif verbosity == 2:
        return InfoLVL1, ProgressBarLVL0
    elif verbosity == 3:
        return InfoLVL1, ProgressBarLVL1
