'''
MIT License

Copyright (c) [2018] [Simon Blanke]
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

import os
import sys
import time
import random
import numpy as np
import pandas as pd
import multiprocessing

from importlib import import_module
from functools import partial
from sklearn.model_selection import cross_val_score

num_cores = multiprocessing.cpu_count()
#print(num_cores, 'CPU threads available')



def random_search(N_searches, X_train, y_train, scoring, ML_dict, cv):
	'''
	In this function we do the random search in the hyperparameter/ML-models space given by the 'ML_dict'-dictionary.
	The goal is to find the model/hyperparameter combination with the best score. This means that we have to train every model on data and compare their scores.

	Arguments:
		- N_searches: Number of model/hyperpara. combinations searched. (int)
		- X_train: training data of features, similar to scikit-learn. (numpy array)
		- y_train: training data of targets, similar to scikit-learn. (numpy array)
		- scoring: scoring used to compare models, similar to scikit-learn. (string)
		- ML_dict: dictionary that contains models and hyperparameter + their ranges and steps for the search. Similar to Tpot package. (dictionary)
		- cv: defines the k of k-fold cross validation. (int)

	Returns:
		- ML_model: A list of model and hyperparameter combinations with best score. (list of scikit-learn objects)
		- score: A list of scores of these models. (list of floats)
	'''	

	#random.seed(N_searches)

	model = None
	hyperpara_values = []
	hyperpara_names = []

	hyperpara_value = []
	hyperpara_name = []
	hyperpara_dict = {}

	dict1_len = len(ML_dict)
	rand1 = random.randint(0, dict1_len-1)

	for i, key_model in zip(range(dict1_len), ML_dict.keys()):
		if i == rand1:
			model = key_model

	if model == None:
		print('Warning: No model selected!\n')

	dict1_len = len(ML_dict[model])
	dict2_len = len(ML_dict[model])
	
	for j, hyperpara_name in zip(range(dict2_len), ML_dict[model].keys()):

		dict3_len = len(ML_dict[model][hyperpara_name])
		rand2 = random.randint(0, dict3_len-1)

		hyperpara_names.append(hyperpara_name)

		array = ML_dict[model][hyperpara_name]
		hyperpara_value = array[rand2]

		hyperpara_values.append(hyperpara_value)

		hyperpara_dict[hyperpara_name] = hyperpara_value


	sklearn, submod_func = model.rsplit('.', 1)
	module = import_module(sklearn)
	model = getattr(module, submod_func)

	ML_model = model(**hyperpara_dict)

	scores = cross_val_score(ML_model, X_train, y_train, scoring=scoring, cv=cv)
	score = scores.mean()

	return ML_model, score



class RandomSearch_Optimizer(object):

	def __init__(self, ML_dict, scoring, N_pipelines=None, T_search_time=None, cv=5, verbosity=0):
		self.ML_dict = ML_dict
		self.scoring = scoring
		self.N_pipelines = N_pipelines
		self.T_search_time = T_search_time
		self.cv = cv
		self.verbosity = verbosity

		self.best_model = None

	def __call__(self):
		return self.best_model
	
	
	def _find_best_model(self, X_train, y_train):
		N_best_models = 1

		models, scores = self._random_search_multiprocessing(X_train, y_train)

		scores = np.array(scores)
		index_best_scores = scores.argsort()[-N_best_models:][::-1]

		best_score = scores[index_best_scores]
		best_pipeline = models[index_best_scores[0]]

		return best_pipeline, best_score
	
	def _random_search_multiprocessing(self, X_train, y_train):
		'''
		This function runs the 'random_search'-function in parallel to return a list of the models and their scores.
		After that the lists are searched to find the best model and its score.

		Arguments:
			- X_train: training data of features, similar to scikit-learn. (numpy array)
			- y_train: training data of targets, similar to scikit-learn. (numpy array)
		Returns:
			- best_pipeline: The model and hyperparameter combination with best score. (scikit-learn object)
			- best_score: The score of this model. (float)
		'''	

		pool = multiprocessing.Pool(num_cores)
		random_search_obj = partial(random_search, ML_dict=self.ML_dict, X_train=X_train, y_train=y_train, scoring=self.scoring, cv=self.cv)
		models, scores = zip(*pool.map(random_search_obj, range(0, self.N_pipelines)))
		pool.close()

		return models, scores


	def fit(self, X_train, y_train):
		self.best_model, best_score = self._find_best_model(X_train=X_train, y_train=y_train)
		self.best_model.fit(X_train, y_train)

		
		
		
		
		
		
		
		
		
		
		
		
		