import time
import datetime
import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor


from discover_meta_knowledge import discover_meta_knowledge
from discover_meta_knowledge import _get_all_features
from discover_meta_knowledge import _save_toHDF
from discover_meta_knowledge import _get_model_score
from discover_meta_knowledge import _get_features_from_model
from discover_meta_knowledge import _get_features_from_dataset

from train_meta_regressor import train_meta_regressor
from train_meta_regressor import _get_meta_knowledge
from train_meta_regressor import _train_regressor

from search_optimum import search_optimum


class Meta_Knowledge_Optimizer(object):
  #-------------------------- discover_meta_knowledge methods --------------------------#
  discover_meta_knowledge = discover_meta_knowledge
  _get_all_features = _get_all_features
  _save_toHDF = _save_toHDF
  _get_model_score = _get_model_score
  _get_features_from_model = _get_features_from_model
  _get_features_from_dataset = _get_features_from_dataset
  
  #-------------------------- train_meta_regressor methods --------------------------#
  train_meta_regressor = train_meta_regressor
  _get_meta_knowledge = _get_meta_knowledge
  _train_regressor = _train_regressor
  
  #-------------------------- search_optimum methods --------------------------#
  search_optimum = search_optimum



  def __init__(self, ml_config_dict, N_models, scoring, cv=5):
    self.ml_config_dict = ml_config_dict
    self.N_models = N_models
    self.scoring = scoring
    self.cv = cv


    self.label_enc = None
    
    ### Datasets
    self.meta_features_df = None
    self.meta_features_df_dict = {}
    
    self.meta_regressor = None
    
   



