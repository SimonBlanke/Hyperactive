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
    self.model = None

    self.X_train = None
    self.y_train = None


  def collect_meta_data(self, dataset_dict, ml_config_dict):

    for data_key in dataset_dict.keys():

      self.X_train = dataset_dict[data_key][0]
      self.y_train = dataset_dict[data_key][1]
      for ml_key in ml_config_dict.keys():
        all_features = self._get_all_features(ml_config_dict)

        self._save_toHDF(all_features, ml_key)


  def _get_features_from_dataset(self):
    def get_number_of_instances():
      return 'N_rows', int(self.X_train.shape[0])

    def get_number_of_features():
      return 'N_columns', int(self.X_train.shape[1])

    def get_default_score():
      return 'cv_default_score', cross_val_score(self.model, self.X_train, self.y_train, cv=5).mean()
      
    # List of functions to get the different features of the dataset
    func_list = [get_number_of_instances, get_number_of_features, get_default_score]
    
    features_from_dataset = {}
    for func in func_list:
      name, value = func()
      features_from_dataset[name] = value
      
    features_from_dataset = pd.DataFrame(features_from_dataset, index=[0])
    features_from_dataset = features_from_dataset.reindex_axis(sorted(features_from_dataset.columns), axis=1)

    return features_from_dataset


  def _import_model(self, model):
    sklearn, submod_func = model.rsplit('.', 1)
    module = import_module(sklearn)
    model = getattr(module, submod_func)

    return model


  def _get_default_hyperpara(self, model, n_rows):
    hyperpara_dict = model.get_params()
    hyperpara_df = pd.DataFrame(hyperpara_dict, index=[0])

    hyperpara_df = pd.DataFrame(hyperpara_df, index=range(n_rows))
    columns = hyperpara_df.columns
    for column in columns:
      hyperpara_df[column] = hyperpara_df[column][0]

    return hyperpara_df


  def _merge_dict(self, params_df, hyperpara_df):
    searched_hyperpara = params_df.columns

    for hyperpara in searched_hyperpara:
      hyperpara_df = hyperpara_df.drop(hyperpara, axis=1)
    params_df = pd.concat([params_df, hyperpara_df], axis=1, ignore_index=False)

    return params_df


  def _grid_search(self, ml_config_dict):
    for model_key in ml_config_dict.keys():
      parameters = ml_config_dict[model_key]

      model = self._import_model(model_key)
      self.model = model()

      model_grid_search = GridSearchCV(self.model, parameters, cv=self.cv, n_jobs=self.n_jobs)
      model_grid_search.fit(self.X_train, self.y_train)

      grid_search_dict = model_grid_search.cv_results_

      params_df = pd.DataFrame(grid_search_dict['params'])

      default_hyperpara_df = self._get_default_hyperpara(self.model, len(params_df))
      params_df = self._merge_dict(params_df, default_hyperpara_df)

      params_df = params_df.reindex_axis(sorted(params_df.columns), axis=1)

      mean_test_score_df = pd.DataFrame(grid_search_dict['mean_test_score'], columns=['mean_test_score'])

      features_from_model = pd.concat([params_df, mean_test_score_df], axis=1, ignore_index=False)

      return features_from_model


  def _get_all_features(self, ml_config_dict):
    features_from_model = self._grid_search(ml_config_dict)
    features_from_dataset = self._get_features_from_dataset()
    

    features_from_dataset = pd.DataFrame(features_from_dataset, index=range(len(features_from_model)))
    columns = features_from_dataset.columns
    for column in columns:
      features_from_dataset[column] = features_from_dataset[column][0]

    all_features = self._concat_dataframes(features_from_dataset, features_from_model)

    #print(all_features.head())

    return all_features


  def _concat_dataframes(self, features_from_dataset, features_from_model):
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
    path1 = './meta_learn/data/'+'meta_knowledge'
    
    #print(key)
    #print(path1)
    #print(len(dataframe))

    dataframe.to_hdf(path1, key='a', mode='a', format='table', append=True)
    #dataframe.to_csv(path1, index=False)
    #print('saving', len(dataframe), 'examples of', str(key), 'meta data')











