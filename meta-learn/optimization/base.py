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

from importlib import import_module
from sklearn.model_selection import cross_val_score


class BaseOptimizer(object):

	def __init__(self):
		pass

	def _check_model_str(self, model):
		if 'sklearn' not in model:
			print(' ')
			return


	def _get_random_hyperparameter(self, ML_dict):
		model = None
		hyperpara_names = []

		hyperpara_dict = {}
		hyperpara_value = []

		model = random.choice(list(ML_dict.keys()))
		self._check_model_str(model)

		for hyperpara_name in ML_dict[model].keys():

			n_hyperpara_values = len(ML_dict[model][hyperpara_name])
			rand_hyperpara_value = random.randint(0, n_hyperpara_values-1)

			hyperpara_names.append(hyperpara_name)

			hyperpara_values = ML_dict[model][hyperpara_name]
			hyperpara_value = hyperpara_values[rand_hyperpara_value]

			hyperpara_dict[hyperpara_name] = hyperpara_value

		return model, hyperpara_dict


	def _import_model(self, model):
		sklearn, submod_func = model.rsplit('.', 1)
		module = import_module(sklearn)
		model = getattr(module, submod_func)

		return model


	def _train_model(self, ML_model, X_train, y_train, scoring, cv):
		train_time = 0
		time_temp = time.time()
		scores = cross_val_score(ML_model, X_train, y_train, scoring=scoring, cv=cv)
		train_time = (time.time() - time_temp)/cv
		score = scores.mean()

		return score, train_time


	def _find_best_model(self, models, scores):
		N_best_models = 1

		scores = np.array(scores)
		index_best_scores = scores.argsort()[-N_best_models:][::-1]

		best_score = scores[index_best_scores]
		best_model = models[index_best_scores[0]]

		return best_model, best_score


	def	get_meta_regressor(model):
		pass


