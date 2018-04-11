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

	scores = cross_val_score(ML_model, X_train, y_train, scoring=self.scoring, cv=self.cv)
	score = scores.mean()

	return score, hyperpara_indices



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

	scores = cross_val_score(ML_model, X_train, y_train, scoring=self.scoring, cv=self.cv)
	score = scores.mean()

	return score, hyperpara_indices_new


def calc_temperature():
	pass




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




class SimulatedAnnealing_Optimizer(object):

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

		self.score = 0
		self.score_best = 0
		self.score_current = 0

		self.hyperpara_indices = 0
		self.hyperpara_indices_best = 0
		self.hyperpara_indices_current = 0



	def fit(self, X_train, y_train):
		calc_general_parameters(self)
		self.score_current, self.hyperpara_indices_current = create_init_solution(self, X_train, y_train)

		self.score_best = self.score_current
		self.hyperpara_indices_best = self.hyperpara_indices_current

		Temp = 100
		for i in range(self.N_pipelines):

			self.score, self.hyperpara_indices = calc_neighbor_solution(self, X_train, y_train, self.hyperpara_indices_current)
			Temp = Temp*0.999

			# Normalized score difference to have a factor for later use with temperature and random
			score_diff_norm = (self.score_current - self.score)/(self.score_current + self.score)
			rand = random.randint(0, 1)

			if self.score > self.score_current:
				self.score_current = self.score
				self.hyperpara_indices_current = self.hyperpara_indices
				if self.score > self.score_best:
					self.score_best = self.score
					self.hyperpara_indices_best = self.hyperpara_indices

			elif np.exp( score_diff_norm/Temp ) > rand:
				self.score_current = self.score
				self.hyperpara_indices_current = self.hyperpara_indices

		print(self.score_best)

		#print(self.hyperpara_length_list)

	def predict(self, X_test):
		pass



	def simulated_annealing(self):
		pass















