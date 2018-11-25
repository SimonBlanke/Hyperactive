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
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler

from .label_encoder_dict import label_encoder_dict


class MetaRegressor(object):

  def __init__(self, model_name):
    self.path = './meta_learn/data/meta_knowledge'
    self.meta_regressor = None

    self.model_name = model_name
    #self._get_model_name()


  def train_meta_regressor(self):
    X_train, y_train = self._get_meta_knowledge()

    X_train = self._label_enconding(X_train)
    #print(X_train)
    self._train_regressor(X_train, y_train)
    self._store_model()


  def _get_model_name(self):
    model_name = self.path.split('/')[-1]
    return model_name


  def _get_hyperpara(self):
    return label_encoder_dict[self.model_name]


  def _label_enconding(self, X_train):
    hyperpara_dict = self._get_hyperpara()

    for hyperpara_key in hyperpara_dict:
      X_train = X_train.replace({str(hyperpara_key): hyperpara_dict[hyperpara_key]})

    return X_train


  def _get_meta_knowledge(self):
    #data = pd.read_csv(self.path)
    data = pd.read_hdf(self.path, key='a')

    #print(data)
    
    column_names = data.columns
    score_name = [name for name in column_names if 'mean_test_score' in name]
    
    X_train = data.drop(score_name, axis=1)
    y_train = data[score_name]

    y_train = self._scale(y_train)

    return X_train, y_train


  def _scale(self, y_train):
    # scale the score -> important for comparison of meta data from datasets in meta regressor training
    scaler = MinMaxScaler()
    y_train = scaler.fit_transform(y_train)
    y_train = pd.DataFrame(y_train, columns=['mean_test_score'])

    return y_train


  def _train_regressor(self, X_train, y_train):
    if self.meta_regressor == None:
      n_estimators = int(y_train.shape[0]/50)
      if n_estimators < 100:
        n_estimators = 100
      if n_estimators > 1000:
        n_estimators = 1000
      n_estimators = 1000
      print('n_estimators: ', n_estimators)

      time1 = time.time()
      #self.meta_regressor = GradientBoostingRegressor(n_estimators=n_estimators)
      self.meta_regressor = xgb.XGBRegressor(n_estimators=n_estimators, nthread=-1)
      print('Meta dataset', y_train.shape[0])
      self.meta_regressor.fit(X_train, y_train)
      print('time: ', round( (time.time() - time1), 4))
    

  def _store_model(self):
    filename = './meta_learn/data/'+str(self.model_name)+'_meta_regressor.pkl'
    #print(filename)
    joblib.dump(self.meta_regressor, filename)









