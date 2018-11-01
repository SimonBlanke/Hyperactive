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

	def __init__(self, verbosity=0):
		self.verbosity = verbosity


	def	get_meta_regressor(model):
		pass


	def _check_model_str(self, model):
		if 'sklearn' not in model:
			print(' ')
			return


	def _get_random_value(self, search_space_dict):
		model = None
		hyperpara_names = []

		hyperpara_dict = {}
		hyperpara_value = []

		model = random.choice(list(search_space_dict.keys()))
		self._check_model_str(model)

		for hyperpara_name in search_space_dict[model].keys():

			n_hyperpara_values = len(search_space_dict[model][hyperpara_name])
			rand_hyperpara_value = random.randint(0, n_hyperpara_values-1)

			hyperpara_names.append(hyperpara_name)

			hyperpara_values = search_space_dict[model][hyperpara_name]
			hyperpara_value = hyperpara_values[rand_hyperpara_value]

			hyperpara_dict[hyperpara_name] = hyperpara_value

		return model, hyperpara_dict


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


	def _search_multiprocessing(self, search_space_dict, function, n_searches):
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

		#partial_function = partial(self._train_model, X_train=X_train, y_train=y_train, scoring=self.scoring, cv=self.cv)
		
		random_search_obj = partial(self._search, search_space_dict=search_space_dict, function=function)
				
		c_time = time.time()
		models, scores, hyperpara_dict, train_time = zip(*pool.map(random_search_obj, range(0, n_searches)))
		
		self.model_list = models[:]
		self.score_list = scores[:]
		self.hyperpara_dict = hyperpara_dict[:]
		self.train_time = train_time[:]

		return models, scores


	def search(self, search_space_dict, function, n_searches):
		models, scores = self._search_multiprocessing(search_space_dict, function, n_searches)

		self.best_model, best_score = self._find_best_model(models, scores)




class machine_learning_helper(object):

	def __init__(self, X_train, y_train, scoring, cv=5):
		self.X_train = X_train
		self.y_train = y_train
		self.scoring = scoring
		self.cv = cv


	def get_partial_ml_function(self):
		partial_function = partial(self._train_model, X_train=self.X_train, y_train=self.y_train, scoring=self.scoring, cv=self.cv)

		return partial_function


	def _train_model(self, ml_model, X_train, y_train, scoring, cv):
		time_temp = time.time()
		scores = cross_val_score(ml_model, X_train, y_train, scoring=scoring, cv=cv)
		train_time = (time.time() - time_temp)/cv

		return scores.mean(), train_time