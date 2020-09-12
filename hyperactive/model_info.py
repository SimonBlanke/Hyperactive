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


def print_results_info(results_list, process_info_dict):
    model_results = {}

    unique_models = []
    for results in results_list:
        nth_process = results["nth_process"]
        model = process_info_dict[nth_process]["model"]

        unique_models.append(model)
    unique_models = set(unique_models)

    for unique_model in unique_models:
        best_score = -np.inf
        best_para = None

        for results in results_list:
            nth_process = results["nth_process"]
            search_space = process_info_dict[nth_process]["search_space"]
            model = process_info_dict[nth_process]["model"]

            if model != unique_model:
                continue

            if results["best_score"] > best_score:
                best_score = results["best_score"]
                pos = results["best_pos"]
                best_para = pos2para(search_space, pos)

        model_results[unique_model] = (best_score, best_para)

    print("\nModel results information:")
    for unique_model in model_results.keys():
        print("   Model name:      ", unique_model.__name__)
        print("   Best score:      ", model_results[unique_model][0])
        print("   Best parameters: ", model_results[unique_model][1])
        print("")
