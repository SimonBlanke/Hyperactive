# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


class Scheduler:
    def __init__(self, n_jobs):
        self.n_jobs = n_jobs


class DefaultScheduler(Scheduler):
    def __init__(self, n_jobs):
        super().__init__(n_jobs)


class SmartScheduler(Scheduler):
    def __init__(self, n_jobs):
        super().__init__(n_jobs)