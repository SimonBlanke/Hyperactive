# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np
import pandas as pd


def expand_dataframe(data_pd, length):
    data_pd = pd.DataFrame(data_pd, index=range(length))
    columns = data_pd.columns
    for column in columns:
        data_pd[column] = data_pd[column][0]

    return data_pd


def merge_meta_data(features_from_dataset, features_from_model):
    length = len(features_from_model)
    features_from_dataset = expand_dataframe(features_from_dataset, length)

    features_from_dataset = features_from_dataset.reset_index()
    features_from_model = features_from_model.reset_index()

    if "index" in features_from_dataset.columns:
        features_from_dataset = features_from_dataset.drop("index", axis=1)
    if "index" in features_from_model.columns:
        features_from_model = features_from_model.drop("index", axis=1)

    all_features = pd.concat(
        [features_from_dataset, features_from_model], axis=1, ignore_index=False
    )

    return all_features


def merge_dict(params_df, hyperpara_df):
    searched_hyperpara = params_df.columns

    for hyperpara in searched_hyperpara:
        hyperpara_df = hyperpara_df.drop(hyperpara, axis=1)
    params_df = pd.concat([params_df, hyperpara_df], axis=1, ignore_index=False)

    return params_df


def get_default_hyperpara(model, n_rows):
    hyperpara_dict = model.get_params()
    hyperpara_df = pd.DataFrame(hyperpara_dict, index=[0])

    hyperpara_df = pd.DataFrame(hyperpara_df, index=range(n_rows))
    columns = hyperpara_df.columns
    for column in columns:
        hyperpara_df[column] = hyperpara_df[column][0]

    return hyperpara_df


def find_best_hyperpara(features, scores):
    N_best_features = 1

    scores = np.array(scores)
    index_best_scores = list(scores.argsort()[-N_best_features:][::-1])

    best_score = scores[index_best_scores][0]
    best_features = features.iloc[index_best_scores]

    return best_features, best_score
