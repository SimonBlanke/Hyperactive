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
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from optimization.random_search import RandomSearch_Optimizer

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor


def discover_meta_knowledge(self, dataset_dict):
  for data_key in dataset_dict.keys():

    X_train = dataset_dict[data_key][0]
    y_train = dataset_dict[data_key][1]
    for ml_key in self.ml_config_dict.keys():
      all_features = self._get_all_features(X_train, y_train)

      
      self.meta_features_df_dict[ml_key] = all_features
      self._save_toHDF(all_features, ml_key)
    # Do data DataCollection
    # Save to file


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


def _get_features_from_model(self, skl_model_obj):
  hyperpara = skl_model_obj.get_params()
  features_from_model = pd.DataFrame(hyperpara, index=[0])

  return features_from_model  


def _get_model_score(self, score):
  score_dict = {}
  #scores = cross_val_score(skl_model_obj, features, target, scoring=scoring, cv=cv)
  name = self.scoring+'___score'
  score_dict[name] = score
  
  score = pd.DataFrame(score_dict, index=[0])
  
  return score  


def _get_all_features(self, X_train, y_train):
  features_from_dataset = self._get_features_from_dataset(X_train)
  
  all_features = pd.DataFrame()

  opt = RandomSearch_Optimizer(ML_dict=self.ml_config_dict, scoring=self.scoring, N_pipelines=self.N_models)
  opt.fit(X_train, y_train)
  
  models = opt.model_list
  scores = opt.score_list
  train_times = opt.train_time
  
  for model, score in zip(models, scores):
    features_from_model = self._get_features_from_model(model)
    score = self._get_model_score(score)
	  
    result = pd.concat([features_from_dataset, features_from_model, score], axis=1)

    all_features = all_features.append(result)

  all_features = all_features.reset_index()
  all_features = all_features.drop('index', axis=1)

  print(all_features)
  # WARNING None becomes NaN during concat. Still apparent in later pandas version?
  return all_features


def _save_toHDF(self, dataframe, key, path=''):
  today = datetime.date.today()
  path_name = path+'meta_knowledge'
  dataframe.to_hdf(path_name, key=str(key), mode='a')
  print(str(key))











