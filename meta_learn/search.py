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

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier



class Search(object):

  def __init__(self, path, search_dict):
    self.path = path
    self.search_dict = search_dict
    self.meta_regressor = None
    self.all_features = None


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


  def _get_features_from_dataset(self, X_train):
    def get_number_of_instances(X_train):
      return 'N_rows', X_train.shape[0]

    def get_number_of_features(X_train):
      return 'N_columns', X_train.shape[1]

    # List of functions to get the different features of the dataset
    func_list = [get_number_of_instances, get_number_of_features]
    
    features_from_dataset = {}
    for func in func_list:
      name, value = func(X_train)
      features_from_dataset[name] = value
      
    features_from_dataset = pd.DataFrame(features_from_dataset, index=[0])
    return features_from_dataset


  def _search(self, X_train):
    for model_key in self.search_dict.keys():

      hyperpara_search_dict = self.search_dict[model_key]

      keys, values = zip(*hyperpara_search_dict.items())
      meta_reg_input = [dict(zip(keys, v)) for v in itertools.product(*values)]

      features_from_model = pd.DataFrame(meta_reg_input)

      features_from_dataset = self._get_features_from_dataset(X_train)

      features_from_dataset = pd.DataFrame(features_from_dataset, index=range(len(features_from_model)))

      columns = features_from_dataset.columns
      for column in columns:
        features_from_dataset[column] = features_from_dataset[column][0]

      features_from_dataset = features_from_dataset.reset_index()
      features_from_model = features_from_model.reset_index()

      if 'index' in features_from_dataset.columns:
        features_from_dataset = features_from_dataset.drop('index', axis=1)
      if 'index' in features_from_model.columns:
        features_from_model = features_from_model.drop('index', axis=1)

      self.all_features = pd.concat([features_from_dataset, features_from_model], axis=1, ignore_index=False)


  def _predict(self):
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

