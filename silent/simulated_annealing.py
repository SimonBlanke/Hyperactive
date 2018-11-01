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

from base import BaseOptimizer


num_cores = multiprocessing.cpu_count()
print(num_cores, 'CPU threads available')


from dict1 import classifier_config_dict
from dict1 import classifier_config_dict1

from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target


def get_model_from_string(module_string, hyperpara_dict):
	sklearn, submod_func = module_string.rsplit('.', 1)
	module = import_module(sklearn)
	model = getattr(module, submod_func)

	ML_model = model(**hyperpara_dict)

	return ML_model


def create_init_solution(self, X_train, y_train):

	model_key = list(self.ML_dict.keys())[0]
	hyperpara_dict = self.ML_dict[model_key]

	hyperpara_dict = {}
	hyperpara_indices = []

	for hyperpara_key in self.ML_dict[model_key]:

		N_values = len(self.ML_dict[model_key][hyperpara_key])
		rand_values = random.randint(0, N_values-1)

		hyperpara_value = self.ML_dict[model_key][hyperpara_key][rand_values]
		hyperpara_dict[hyperpara_key] = hyperpara_value
		hyperpara_indices.append(rand_values)

	ML_model = get_model_from_string(model_key, hyperpara_dict)
	ML_model_str = str(ML_model)

	scores = cross_val_score(ML_model, X_train, y_train, scoring=self.scoring, cv=self.cv)
	score = scores.mean()

	return score, hyperpara_indices, hyperpara_dict, ML_model_str



def calc_neighbor_solution(self, X_train, y_train, hyperpara_indices, epsilon=1):
	Random_list = []

	N_hyperparameters = len(hyperpara_indices)
	
	for i in range(N_hyperparameters):
		Random_list.append(random.randint(-epsilon, epsilon))

	Random_list = np.array(Random_list)

	hyperpara_indices_new = np.add(hyperpara_indices, Random_list)

	hyperpara_length_np = np.array(self.hyperpara_length_list)

	hyperpara_indices_new[hyperpara_indices_new < 0] = 0
	hyperpara_indices_new[hyperpara_indices_new+1 > hyperpara_length_np] = hyperpara_length_np[hyperpara_indices_new+1 > hyperpara_length_np]-1

	model_key = list(self.ML_dict.keys())[0]
	hyperpara_dict = {}
	for hyperpara_key, i in zip(self.ML_dict[model_key], range(len(self.ML_dict[model_key]))):

		out = list(self.ML_dict[model_key][hyperpara_key])[hyperpara_indices_new[i]]

		hyperpara_dict[hyperpara_key] = out




	ML_model = get_model_from_string(model_key, hyperpara_dict)
	ML_model_str = str(ML_model)

	scores = cross_val_score(ML_model, X_train, y_train, scoring=self.scoring, cv=self.cv)
	score = scores.mean()

	return score, hyperpara_indices_new, hyperpara_dict, ML_model_str



def calc_general_parameters(self):
	self.model_key = list(self.ML_dict.keys())[0]
	self.hyperpara_list = list(self.ML_dict[self.model_key].keys())

	self.hyperpara_range_list = []
	self.hyperpara_length_list = []


	for hyperpara in self.hyperpara_list:
		hyperpara_range = np.array(self.ML_dict[self.model_key][hyperpara])
		hyperpara_length = len(hyperpara_range)
		self.hyperpara_range_list.append(hyperpara_range)
		self.hyperpara_length_list.append(hyperpara_length)




class SimulatedAnnealing_Optimizer(BaseOptimizer):

	calc_general_parameters = calc_general_parameters
	create_init_solution = create_init_solution
	calc_neighbor_solution = calc_neighbor_solution


	def __init__(self, ML_dict, scoring, N_pipelines=None, T_search_time=None, cv=5, verbosity=0):
		self.ML_dict = ML_dict
		self.scoring = scoring
		self.N_pipelines = N_pipelines
		self.T_search_time = T_search_time
		self.cv = cv
		self.verbosity = verbosity

		self.best_model = None

		self.score = 0
		self.score_best = 0
		self.score_current = 0

		self.hyperpara_indices = 0
		self.hyperpara_indices_best = 0
		self.hyperpara_indices_current = 0

		self.hyperpara_dict = 0
		self.hyperpara_dict_best = 0
		self.hyperpara_dict_current = 0

		self.ML_model_str = 0
		self.ML_model_str_best = 0
		self.ML_model_str_current = 0

	def __call__(self):
		return self.best_model


	def fit(self, X_train, y_train):
		calc_general_parameters(self)
		self.score_current, self.hyperpara_indices_current, self.hyperpara_dict_current, self.ML_model_str_current = create_init_solution(self, X_train, y_train)

		self.score_best = self.score_current
		self.hyperpara_indices_best = self.hyperpara_indices_current
		self.hyperpara_dict_best = self.hyperpara_dict_current
		self.ML_model_str_best = self.ML_model_str_current

		Temp = 100
		for i in range(self.N_pipelines):

			self.score, self.hyperpara_indices, self.hyperpara_dict, self.ML_model_str  = calc_neighbor_solution(self, X_train, y_train, self.hyperpara_indices_current)
			Temp = Temp*0.999

			# Normalized score difference to have a factor for later use with temperature and random
			score_diff_norm = (self.score_current - self.score)/(self.score_current + self.score)
			rand = random.randint(0, 1)

			if self.score > self.score_current:
				self.score_current = self.score
				self.hyperpara_indices_current = self.hyperpara_indices
				self.hyperpara_dict_current = self.hyperpara_dict
				self.ML_model_str_current = self.ML_model_str 

				if self.score > self.score_best:
					self.score_best = self.score
					self.hyperpara_indices_best = self.hyperpara_indices
					self.hyperpara_dict_best = self.hyperpara_dict
					self.ML_model_str_best  = self.ML_model_str 

			elif np.exp( score_diff_norm/Temp ) > rand:
				self.score_current = self.score
				self.hyperpara_indices_current = self.hyperpara_indices
				self.hyperpara_dict_current = self.hyperpara_dict
				self.ML_model_str_current = self.ML_model_str 


		self.best_model = get_model_from_string(self.ML_model_str_best, self.hyperpara_dict_best)
		

