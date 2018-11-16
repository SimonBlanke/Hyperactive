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

from importlib import import_module

from sklearn.datasets import load_iris
from sklearn.datasets import load_wine

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor


class DataCollector(object):

  def __init__(self, scoring, cv=5, n_jobs=-1):
    self.scoring = scoring
    self.cv = cv
    self.n_jobs = n_jobs


  def collect_meta_data(self, dataset_dict, ml_config_dict):

    for data_key in dataset_dict.keys():

      X_train = dataset_dict[data_key][0]
      y_train = dataset_dict[data_key][1]
      for ml_key in ml_config_dict.keys():
        all_features = self._get_all_features(X_train, y_train, ml_config_dict)

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


  def _import_model(self, model):
    sklearn, submod_func = model.rsplit('.', 1)
    module = import_module(sklearn)
    model = getattr(module, submod_func)

    return model


  def _grid_search(self, X_train, y_train, ml_config_dict):
    for model_key in ml_config_dict.keys():
      parameters = ml_config_dict[model_key]

      model = self._import_model(model_key)
      model = model()

      model_grid_search = GridSearchCV(model, parameters, cv=self.cv, n_jobs=self.n_jobs)
      model_grid_search.fit(X_train, y_train)

      grid_search_dict = model_grid_search.cv_results_

      #print(grid_search_dict.keys(), '\n')

      params_df = pd.DataFrame(grid_search_dict['params'])
      mean_test_score_df = pd.DataFrame(grid_search_dict['mean_test_score'], columns=['mean_test_score'])

      #print(params_df, '\n')
      #print(mean_test_score_df, '\n')

      features_from_model = pd.concat([params_df, mean_test_score_df], axis=1, ignore_index=False)
      #print(features_from_model, '\n')

      return features_from_model


  def _get_all_features(self, X_train, y_train, ml_config_dict):
    features_from_dataset = self._get_features_from_dataset(X_train)
    
    features_from_model = self._grid_search(X_train, y_train, ml_config_dict)


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

    all_features = pd.concat([features_from_dataset, features_from_model], axis=1, ignore_index=False)
 
    return all_features


  def _save_toHDF(self, dataframe, key, path=''):
    today = datetime.date.today()
    path_name = path+'meta_knowledge'
    #print(dataframe)
    #dataframe.to_hdf(path_name, key=str(key), mode='a')
    dataframe.to_csv(key, index=False)
    print('saving', str(key), 'meta data')











