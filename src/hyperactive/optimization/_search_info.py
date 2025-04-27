from dataclasses import dataclass


@dataclass()
class SearchInfo:
    experiment: int
    s_space: int
    n_iter: int
    initialize: int
    constraints: int
    max_score: int
    early_stopping: int
    random_state: int
    memory: int
    memory_warm_start: int
