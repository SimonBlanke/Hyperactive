from dataclasses import dataclass


@dataclass()
class RunInfo:
    max_time: int
    distribution: int
    n_processes: int
    verbosity: int
