# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np


class SearchInfo:
    def __init__(self):
        self.search_infos = []

    def add_search(
        self,
        processes,
        model,
        search_space,
        n_iter,
        name,
        optimizer,
        n_jobs,
        initialize,
        memory,
    ):

        space_lengths = []
        for dict_value in search_space.values():
            space_len = len(dict_value)
            space_lengths.append(space_len)

        space_lengths = np.array(space_lengths)
        search_space_size = np.prod(space_lengths)

        self.search_infos.append(
            {
                "processes": processes,
                "model": model,
                "search_space": search_space,
                "search_space_size": search_space_size,
                "n_iter": n_iter,
                "name": name,
                "optimizer": optimizer,
                "n_jobs": n_jobs,
                "initialize": initialize,
                "memory": memory,
            }
        )

    def print_search_info(self):
        print("\nSearch information:")
        for search_info in self.search_infos:
            print("   Processes:                ", *search_info["processes"])
            print("   Model name:               ", search_info["model"].__name__)
            print("   Optimization strategy:    ", search_info["optimizer"])
            print("   Search space:")
            print("      size:                  ", search_info["search_space_size"])
            print("      known:                 ", 1)
            print("")

