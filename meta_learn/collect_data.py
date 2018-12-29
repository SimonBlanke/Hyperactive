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
from tqdm import tqdm


from sklearn.datasets import load_iris
from sklearn.datasets import load_wine

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor

from .augment_datasets import augment_dataset




class DataCollector(object):

  def __init__(self, scoring, cv=5, n_jobs=-1):
    self.scoring = scoring
    self.cv = cv
    self.n_jobs = n_jobs

    self.dataset_dict = None
    self.ml_config_dict = None

    self.model = None

    self.X_train = None
    self.y_train = None
    self.data_name = None

    self.model_name = None

    self.meta_knowledge_dict = {}


  def collect_meta_data(self, dataset_dict, ml_config_dict, augment_data=False):
    self.dataset_dict = dataset_dict
    self.ml_config_dict = ml_config_dict

    '''
    if augment_data == True:
      dataset_dict_temp = {}
      print('Augmenting data')
      for data_key in tqdm(dataset_dict.keys()):
        X_train = dataset_dict[data_key][0]
        y_train = dataset_dict[data_key][1]

        dataset_dict_t = augment_dataset(X_train, y_train, n_drops=1)

        for key in dataset_dict_t:
          dataset_dict_temp[key] = dataset_dict_t[key]

      dataset_dict = dict(dataset_dict_temp)
    '''

    print('Collecting meta data')

    for model_name in ml_config_dict.keys():
      dataCollector_model = DataCollector_model(ml_config_dict)
      dataCollector_dataset = DataCollector_dataset(ml_config_dict)

      for data_name, data_train in tqdm(dataset_dict.items()):

        features_from_model = dataCollector_model.collect(data_name, data_train)
        features_from_dataset = dataCollector_dataset.collect(data_name, data_train)

        print('features_from_model\n', features_from_model, '\n')
        print('features_from_dataset\n', features_from_dataset, '\n')

        features_from_dataset = pd.DataFrame(features_from_dataset, index=range(len(features_from_model)))
        columns = features_from_dataset.columns
        for column in columns:
          features_from_dataset[column] = features_from_dataset[column][0]
        

        all_features = self._concat_dataframes(features_from_dataset, features_from_model)

        self._save_toHDF(all_features, model_name)


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

    #dataframe.to_hdf(path1, key='a', mode='a', format='table', append=True)
    dataframe.to_csv(path1, index=False)
    #print('saving', len(dataframe), 'examples of', str(key), 'meta data')



from .dataset_features import add_dataset_name
from .dataset_features import get_number_of_instances
from .dataset_features import get_number_of_features
from .dataset_features import get_default_score

class DataCollector_dataset(object):

  add_dataset_name = add_dataset_name
  get_number_of_instances = get_number_of_instances
  get_number_of_features = get_number_of_features
  get_default_score = get_default_score

  def __init__(self, ml_config_dict, cv=5):
    self.ml_config_dict = ml_config_dict

    self.model_name = list(ml_config_dict.keys())[0]


  def _get_features_from_dataset(self):
      
    # List of functions to get the different features of the dataset
    func_list = [self.add_dataset_name, self.get_number_of_instances, self.get_number_of_features, self.get_default_score]
    
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


  def collect(self, data_name, data_train):
    self.data_name = data_name
    self.X_train = data_train[0]
    self.y_train = data_train[1]

    model = self._import_model(self.model_name)
    self.model = model()

    features_from_dataset = self._get_features_from_dataset()

    return features_from_dataset




class DataCollector_model(object):

  def __init__(self, ml_config_dict, cv=5, n_jobs=-1):
    self.ml_config_dict = ml_config_dict
    self.cv = cv
    self.n_jobs = n_jobs


  def _grid_search(self, X_train, y_train):
    for model_name in self.ml_config_dict.keys():
      parameters = self.ml_config_dict[model_name]

      model = self._import_model(model_name)
      self.model = model()

      model_grid_search = GridSearchCV(self.model, parameters, cv=self.cv, n_jobs=self.n_jobs)
      model_grid_search.fit(X_train, y_train)

      grid_search_dict = model_grid_search.cv_results_

      params_df = pd.DataFrame(grid_search_dict['params'])

      default_hyperpara_df = self._get_default_hyperpara(self.model, len(params_df))
      params_df = self._merge_dict(params_df, default_hyperpara_df)

      params_df = params_df.reindex_axis(sorted(params_df.columns), axis=1)

      mean_test_score_df = pd.DataFrame(grid_search_dict['mean_test_score'], columns=['mean_test_score'])

      features_from_model = pd.concat([params_df, mean_test_score_df], axis=1, ignore_index=False)

      return features_from_model


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


  def _import_model(self, model):
    sklearn, submod_func = model.rsplit('.', 1)
    module = import_module(sklearn)
    model = getattr(module, submod_func)

    return model


  def collect(self, data_name, data_train):
    self.data_name = data_name
    X_train = data_train[0]
    y_train = data_train[1]

    return self._grid_search(X_train, y_train)