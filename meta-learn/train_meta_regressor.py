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

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict


def train_meta_regressor(self, path='meta_knowledge'):
  X_train, y_train = self._get_meta_knowledge(path)
  
  self._train_regressor(X_train, y_train)


def _get_meta_knowledge(self, path):
  data = pd.read_hdf(path, key='sklearn.ensemble.RandomForestClassifier')
  
  column_names = data.columns
  score_name = [name for name in column_names if '___score' in name]
  
  X_train = data.drop(score_name, axis=1)
  y_train = data[score_name]

  #print(X_train)

  label_enc = defaultdict(LabelEncoder)
  X_train_trafo = X_train.apply(lambda x: label_enc[x.name].fit_transform(x))
  self.label_enc = label_enc

  print(X_train_trafo)

  X_train = X_train_trafo.apply(lambda x: label_enc[x.name].inverse_transform(x))

  #print(X_train)

  return X_train_trafo, y_train


def _train_regressor(self, X_train, y_train):
  if self.meta_regressor == None:
    self.meta_regressor = GradientBoostingRegressor()
    self.meta_regressor.fit(X_train, y_train)
  

def _store_model(self, model):
  filename = 'blablabla'
  _ = joblib.dump(model, filename, compress=9)  









