# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np


def pos2para(search_space, pos):
    values_dict = {}
    for i, key in enumerate(search_space.keys()):
        pos_ = int(pos[i])
        values_dict[key] = search_space[key][pos_]

    return values_dict


def print_results_info(results_list):
    model_results = {}

    unique_model_names = []
    for results in results_list:
        unique_model_names.append(results["model_name"])
    unique_model_names = set(unique_model_names)

    for model_name in unique_model_names:
        best_score = -np.inf
        best_para = None

        for results in results_list:
            if results["model_name"] != model_name:
                continue

            if results["best_score"] > best_score:
                best_score = results["best_score"]

                pos = results["best_pos"]
                best_para = pos2para(results["search_space"], pos)

        model_results[model_name] = (best_score, best_para)

    print("\nModel results information:")
    for model_name in model_results.keys():
        print("   Model name:      ", model_name)
        print("   Best score:      ", model_results[model_name][0])
        print("   Best parameters: ", model_results[model_name][1])
        print("")
