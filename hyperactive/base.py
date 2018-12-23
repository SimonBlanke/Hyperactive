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

import os
import sys
import time
import random
import numpy as np
import pandas as pd
import multiprocessing

num_cores = multiprocessing.cpu_count()

from importlib import import_module
from functools import partial
from sklearn.model_selection import cross_val_score


class BaseOptimizer(object):

	def __init__(self, ml_search_dict, n_searches, scoring, cv=5, verbosity=0):
		self.ml_search_dict = ml_search_dict
		self.n_searches = n_searches
		self.scoring = scoring
		self.cv = cv
		self.verbosity = verbosity

		self.sklearn_model = None


	def _check_model_str(self, model):
		if 'sklearn' not in model:
			if self.sklearn_model:
				print('')
				return self.sklearn_model
				
			print('No sklearn model in ml_search_dict')
			return
		return model


# score, hyperpara_indices, hyperpara_dict, ML_model_str

	def _get_random_position(self, search_space_dict):
		'''
		get a random N-Dim position in search space and return: 
		model (string), 
		hyperparameter (dict), 
		N indices of N-Dim position (dict)
		'''
		hyperpara_dict = {}
		hyperpara_indices = {}

		# if there are multiple models, select a random one
		model = random.choice(list(search_space_dict.keys()))
		model = self._check_model_str(model)

		for hyperpara_name in search_space_dict[model].keys():

			n_hyperpara_values = len(search_space_dict[model][hyperpara_name])
			hyperpara_index = random.randint(0, n_hyperpara_values-1)

			hyperpara_values = search_space_dict[model][hyperpara_name]
			hyperpara_value = hyperpara_values[hyperpara_index]

			hyperpara_dict[hyperpara_name] = hyperpara_value
			hyperpara_indices[hyperpara_name] = hyperpara_index

		return model, hyperpara_dict, hyperpara_indices


	def _import_model(self, model):
		sklearn, submod_func = model.rsplit('.', 1)
		module = import_module(sklearn)
		model = getattr(module, submod_func)

		return model


	def _find_best_model(self, models, scores):
		N_best_models = 1

		scores = np.array(scores)
		index_best_scores = scores.argsort()[-N_best_models:][::-1]

		best_score = scores[index_best_scores]
		best_model = models[index_best_scores[0]]

		return best_model, best_score


	def _train_model(self, sklearn_model, X_train, y_train):
		time_temp = time.time()
		scores = cross_val_score(sklearn_model, X_train, y_train, scoring=self.scoring, cv=self.cv)
		train_time = (time.time() - time_temp)/self.cv

		return scores.mean(), train_time


	def _search_multiprocessing(self, X_train, y_train, init_search_dict):
		'''
		This function runs the 'random_search'-function in parallel to return a list of the models and their scores.
		After that the lists are searched to find the best model and its score.
		Arguments:
			- X_train: training data of features, similar to scikit-learn. (numpy array)
			- y_train: training data of targets, similar to scikit-learn. (numpy array)
		Returns:
			- best_model: The model and hyperparameter combination with best score. (scikit-learn object)
			- best_score: The score of this model. (float)
		'''	

		pool = multiprocessing.Pool(num_cores)
	
		search = partial(self._search, X_train=X_train, y_train=y_train, ml_search_dict=self.ml_search_dict, init_search_dict=init_search_dict)
				
		c_time = time.time()
		models, scores, hyperpara_dict, train_time = zip(*pool.map(search, range(0, self.n_searches)))
		
		self.model_list = models[:]
		self.score_list = scores[:]
		self.hyperpara_dict = hyperpara_dict[:]
		self.train_time = train_time[:]

		return models, scores


	def fit(self, X_train, y_train, init_search_dict=None):
		models, scores = self._search_multiprocessing(X_train, y_train, init_search_dict)

		self.best_model, best_score = self._find_best_model(models, scores)
		self.best_model.fit(X_train, y_train)

		print('Best score:', *best_score)


	def predict(self, X_test):
		return self.best_model.predict(X_test)


	def score():
		pass
