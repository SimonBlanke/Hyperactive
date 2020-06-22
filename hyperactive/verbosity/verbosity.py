from .info import InfoLVL0, InfoLVL1
from .progress_bar import ProgressBar, ProgressBarLVL0, ProgressBarLVL1


class Verbosity:
    def __init__(self, verbosity, warnings):
        info, p_bar = self._get_verb_classes(verbosity)
        self.info = info()
        self.p_bar = p_bar()

    def _get_verb_classes(self, verbosity):
        if verbosity == 0:
            return InfoLVL0, ProgressBar
        elif verbosity == 1:
            return InfoLVL1, ProgressBar
        elif verbosity == 2:
            return InfoLVL1, ProgressBarLVL0
        elif verbosity == 3:
            return InfoLVL1, ProgressBarLVL1
