'''
MIT License
Copyright (c) [2018] [Simon Franz Albert Blanke]
Email: simonblanke528481@gmail.com
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import time
import datetime
import pickle
import itertools
import numpy as np
import pandas as pd

from functools import partial
from scipy.optimize import minimize
from sklearn.externals import joblib
from importlib import import_module

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from .collect_data import DataCollector
from .meta_regressor import MetaRegressor
from .label_encoder_dict import label_encoder_dict


class Search(DataCollector, MetaRegressor):

  def __init__(self, search_dict):
    self.path = './meta_learn/data/sklearn.neighbors.KNeighborsClassifier_meta_regressor'
    self.search_dict = search_dict
    self.meta_regressor = None
    self.all_features = None

    self.model_name = None


  def search_optimum(self, X_train, y_train):
    time1 = time.time()
    self._load_model()
    print('\n Time _load_model:', time.time() - time1)

    time1 = time.time()
    self._search(X_train)
    print('\n Time _search:', time.time() - time1)

    time1 = time.time()    
    hyperpara_dict, best_score = self._predict()
    print('\n Time _predict:', time.time() - time1, '\n')

    return hyperpara_dict, best_score


  def _load_model(self):
    reg = joblib.load(self.path)
    self.meta_regressor = reg


  def _search(self, X_train):
    for model_key in self.search_dict.keys():
      self.model_name = model_key

      model = self._import_model(model_key)
      model = model()

      hyperpara_search_dict = self.search_dict[model_key]

      keys, values = zip(*hyperpara_search_dict.items())
      meta_reg_input = [dict(zip(keys, v)) for v in itertools.product(*values)]

      features_from_model = pd.DataFrame(meta_reg_input)

      default_hyperpara_df = self._get_default_hyperpara(model, len(features_from_model))
      features_from_model = self._merge_dict(features_from_model, default_hyperpara_df)

      features_from_dataset = self._get_features_from_dataset(X_train)

      features_from_dataset = pd.DataFrame(features_from_dataset, index=range(len(features_from_model)))

      columns = features_from_dataset.columns
      for column in columns:
        features_from_dataset[column] = features_from_dataset[column][0]

      self.all_features = self._concat_dataframes(features_from_dataset, features_from_model)


  def _predict(self):
    self.all_features = self._label_enconding(self.all_features)
    score_pred = self.meta_regressor.predict(self.all_features)

    best_features, best_score = self._find_best_hyperpara(self.all_features, score_pred)

    list1 = list(self.search_dict[str(*self.search_dict.keys())].keys())

    keys = list(best_features[list1].columns)
    values = list(*best_features[list1].values)
    hyperpara_dict = dict(zip(keys, values))

    return hyperpara_dict, best_score
   

  def _find_best_hyperpara(self, features, scores):
    N_best_features = 1

    scores = np.array(scores)
   
    index_best_scores = list(scores.argsort()[-N_best_features:][::-1])

    best_score = scores[index_best_scores][0]
    best_features = features.iloc[index_best_scores]

    return best_features, best_score

