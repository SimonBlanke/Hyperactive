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

from tqdm import tqdm
from pathlib import Path
from functools import partial
from scipy.optimize import minimize
from sklearn.externals import joblib
from importlib import import_module

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from .collect_data import DataCollector
from .collect_data import DataCollector_model
from .collect_data import DataCollector_dataset

from .meta_regressor import MetaRegressor

from .label_encoder_dict import label_encoder_dict


class Search(DataCollector, DataCollector_model, MetaRegressor):

  def __init__(self, search_dict):
    self.path = None
    self.search_dict = search_dict
    self.meta_regressor = None
    self.meta_knowledge = None

    self.model_name = None
    self.hyperpara_search_dict = None

    self._get_meta_regressor_path()

    self.model = None
    self.X_train = None
    self.y_train = None

    self.data_name = None
    self.data_test = None

    self.hyperpara_dict = None


  def search_optimum(self, data_dict):
    for data_name, data_test in tqdm(data_dict.items()):
      self.data_name = data_name
      self.data_test = data_test

      self.X_train = data_test[0]
      self.y_train = data_test[1]

      self._load_model()

      if not self.meta_regressor:
        print('Error: No meta regressor loaded\n')
        return 0

      self._search(self.X_train)
      hyperpara_dict, best_score = self._predict()

      return hyperpara_dict, best_score


  def _load_model(self):
    if Path(self.path).exists():
      reg = joblib.load(self.path)
      self.meta_regressor = reg
    else:
      print('No proper meta regressor found\n')


  def _get_meta_regressor_path(self):
    model_key = self.search_dict.keys()
    path_dir = './data/'
    self.path = path_dir+str(*model_key)+'_meta_regressor.pkl'


  def _search(self, X_train):
    for model_key in self.search_dict.keys():
      self.model_name = model_key

      self.hyperpara_dict = self._get_hyperpara(model_key)

      model = self._import_model(model_key)
      self.model = model()

      self.hyperpara_search_dict = self.search_dict[model_key]

      if len(self.hyperpara_search_dict.items()) == 0:
        print('Error: hyperparameter search dict is empty\n')
        break

      features_from_model = self._features_from_model()

      dataCollector_dataset = DataCollector_dataset(self.search_dict)
      features_from_dataset = dataCollector_dataset.collect(self.data_name, self.data_test)

      features_from_dataset = pd.DataFrame(features_from_dataset, index=range(len(features_from_model)))

      columns = features_from_dataset.columns
      for column in columns:
        features_from_dataset[column] = features_from_dataset[column][0]

      self.meta_knowledge = self._concat_dataframes(features_from_dataset, features_from_model)


  def _features_from_model(self):
    keys, values = zip(*self.hyperpara_search_dict.items())
    meta_reg_input = [dict(zip(keys, v)) for v in itertools.product(*values)]

    features_from_model = pd.DataFrame(meta_reg_input)

    default_hyperpara_df = self._get_default_hyperpara(self.model, len(features_from_model))
    features_from_model = self._merge_dict(features_from_model, default_hyperpara_df)
    features_from_model = features_from_model.reindex_axis(sorted(features_from_model.columns), axis=1)

    return features_from_model

  def _predict(self):
    self.meta_knowledge = self._label_enconding(self.meta_knowledge)
    print(self.meta_knowledge)
    print(self.meta_knowledge.info())
    score_pred = self.meta_regressor.predict(self.meta_knowledge)

    best_features, best_score = self._find_best_hyperpara(self.meta_knowledge, score_pred)

    list1 = list(self.search_dict[str(*self.search_dict.keys())].keys())

    keys = list(best_features[list1].columns)
    values = list(*best_features[list1].values)
    best_hyperpara_dict = dict(zip(keys, values))

    best_hyperpara_dict = self._decode_hyperpara_dict(best_hyperpara_dict)

    return best_hyperpara_dict, best_score


  def _decode_hyperpara_dict(self, hyperpara_dict):
    for hyperpara_key in label_encoder_dict[self.model_name]:
      if hyperpara_key in hyperpara_dict:
        inv_label_encoder_dict = {v: k for k, v in label_encoder_dict[self.model_name][hyperpara_key].items()}

        encoded_values = hyperpara_dict[hyperpara_key]

        hyperpara_dict[hyperpara_key] = inv_label_encoder_dict[encoded_values]

    return hyperpara_dict
   

  def _find_best_hyperpara(self, features, scores):
    N_best_features = 1

    scores = np.array(scores)
    index_best_scores = list(scores.argsort()[-N_best_features:][::-1])

    best_score = scores[index_best_scores][0]
    best_features = features.iloc[index_best_scores]

    return best_features, best_score


  def _get_hyperpara(self, model_name):
    return label_encoder_dict[model_name]


  def _label_enconding(self, X_train):
    for hyperpara_key in self.hyperpara_dict:
      to_replace = {hyperpara_key: self.hyperpara_dict[hyperpara_key] }
      X_train = X_train.replace(to_replace)

    return X_train