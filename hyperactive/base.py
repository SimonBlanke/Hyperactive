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

from importlib import import_module
from functools import partial
from sklearn.model_selection import cross_val_score


class BaseOptimizer(object):

	def __init__(self, ml_search_dict, n_searches, scoring, n_jobs=1, cv=5, verbosity=0):
		self.ml_search_dict = ml_search_dict
		self.n_searches = n_searches
		self.scoring = scoring
		self.n_jobs = n_jobs
		self.cv = cv
		self.verbosity = verbosity

		self.X_train = None
		self.y_train = None
		self.init_search_dict = None

		self.model_key = list(self.ml_search_dict.keys())[0]
		#model_str = random.choice(list(self.ml_search_dict.keys()))
		self.hyperpara_search_dict = ml_search_dict[list(ml_search_dict.keys())[0]]
		

		self._set_n_jobs()
		self.n_searches_range = range(0, self.n_jobs)

		self._check_sklearn_model_key()


	def _set_n_jobs(self):
		num_cores = multiprocessing.cpu_count()
		if self.n_jobs == -1 or self.n_jobs > num_cores:
			self.n_jobs = num_cores
		if self.n_jobs > self.n_searches:
			self.n_searches = self.n_jobs


	def _check_sklearn_model_key(self):
		if 'sklearn' not in self.model_key:		
			raise ValueError('No sklearn model in ml_search_dict found')


	def _get_random_position(self):
		'''
		get a random N-Dim position in search space and return: 
		model (string), 
		hyperparameter (dict), 
		N indices of N-Dim position (dict)
		'''
		search_positions = {}
		
		for hyperpara_name in self.hyperpara_search_dict.keys():

			n_hyperpara_values = len(self.hyperpara_search_dict[hyperpara_name])
			search_position = random.randint(0, n_hyperpara_values-1)

			hyperpara_values = self.hyperpara_search_dict[hyperpara_name]
			hyperpara_value = hyperpara_values[search_position]

			search_positions[hyperpara_name] = search_position

		return search_positions


	def _get_hyperpara_dict_from_positions(self, search_positions):
		hyperpara_dict = {}

		for hyperpara_name in search_positions.keys():
			search_position = search_positions[hyperpara_name]
			hyperpara_dict[hyperpara_name] = list(self.hyperpara_search_dict[hyperpara_name])[search_position]

		return hyperpara_dict


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


	def _create_sklearn_model(self, model, hyperpara_dict):
		return model(**hyperpara_dict)


	def _train_model(self, sklearn_model):
		time_temp = time.time()
		scores = cross_val_score(sklearn_model, self.X_train, self.y_train, scoring=self.scoring, cv=self.cv)
		train_time = (time.time() - time_temp)/self.cv

		return scores.mean(), train_time


	def _get_score(self, hyperpara_dict):
		model = self._import_model(self.model_key)
		sklearn_model = self._create_sklearn_model(model, hyperpara_dict)
		score, train_time = self._train_model(sklearn_model)

		return score, train_time, sklearn_model


	def _search_multiprocessing(self):
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
		
		pool = multiprocessing.Pool(self.n_jobs)		
		models, scores, hyperpara_dict, train_time = zip(*pool.map(self._search, self.n_searches_range))
		
		self.model_list = models
		self.score_list = scores
		self.hyperpara_dict = hyperpara_dict
		self.train_time = train_time

		return models, scores


	def fit(self, X_train, y_train, init_search_dict=None):
		self.X_train = X_train
		self.y_train = y_train
		self.init_search_dict = init_search_dict

		models, scores = self._search_multiprocessing()

		self.best_model, best_score = self._find_best_model(models, scores)
		self.best_model.fit(X_train, y_train)

		print('Best score:', *best_score)


	def predict(self, X_test):
		return self.best_model.predict(X_test)


	def score():
		pass
