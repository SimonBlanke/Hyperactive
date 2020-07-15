# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


class Info:
    def warm_start(self):
        pass

    def scatter_start(self):
        pass

    def random_start(self):
        pass

    def load_meta_data(self):
        pass

    def no_meta_data(self, model_func):
        pass

    def load_samples(self, para):
        pass


class InfoLVL0(Info):
    def __init__(self):
        pass

    def print_start_point(self):
        pass
        # return cand._get_warm_start()


class InfoLVL1(InfoLVL0):
    def __init__(self):
        pass

    def print_start_point(self):
        pass
        # print("best para =", start_point)
        # print("score     =", score_best, "\n")

    def warm_start(self):
        print("Set warm start")

    def scatter_start(self):
        print("Set scatter init")

    def random_start(self):
        print("Set random start position")

    def load_meta_data(self):
        print("Loading meta data successful", end="\r")

    def no_meta_data(self, model_func):
        print("No meta data found for", model_func.__name__, "function")

    def load_samples(self, para):
        print("Loading meta data successful:", len(para), "samples found")
