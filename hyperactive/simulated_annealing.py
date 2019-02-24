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

from .base import BaseOptimizer


class SimulatedAnnealing_Optimizer(BaseOptimizer):

	def __init__(self, ml_search_dict, n_searches, scoring, epsilon=1, annealing_factor=0.9, n_jobs=1, cv=5, verbosity=0):
		super().__init__(ml_search_dict, n_searches, scoring, n_jobs, cv, verbosity)

		self._search = self._start_simulated_annealing

		self.ml_search_dict = ml_search_dict
		self.scoring = scoring
		self.n_searches = n_searches

		self.epsilon = epsilon
		self.annealing_factor = annealing_factor

		self.temp = 100

		self.best_model = None

		self.score = 0
		self.score_best = 0
		self.score_current = 0

		self.hyperpara_indices = 0
		self.hyperpara_indices_best = 0
		self.hyperpara_indices_current = 0


	def _get_neighbor_model(self, hyperpara_indices):
		hyperpara_indices_new = {}

		for hyperpara_name in hyperpara_indices:
			n_values = len(self.hyperpara_search_dict[hyperpara_name])
			rand_epsilon = random.randint(-self.epsilon, self.epsilon+1)

			index = hyperpara_indices[hyperpara_name]
			index_new = index + rand_epsilon
			hyperpara_indices_new[hyperpara_name] = index_new

			# don't go out of range
			if index_new < 0:
				index_new = 0
			if index_new > n_values-1:
				index_new = n_values-1

			hyperpara_indices_new[hyperpara_name] = index_new

		return hyperpara_indices_new


	def _start_simulated_annealing(self, n_searches):
		n_steps = int(self.n_searches/self.n_jobs)

		self.hyperpara_indices_current = self._get_random_position()
		self.hyperpara_dict_current = self._get_hyperpara_dict_from_positions(self.hyperpara_indices_current)
		self.score_current, train_time, sklearn_model = self._get_score(self.hyperpara_dict_current)

		self.score_best = self.score_current
		self.hyperpara_indices_best = self.hyperpara_indices_current

		for i in range(n_steps):
			self.temp = self.temp*self.annealing_factor
			rand = random.randint(0, 1)

			self.hyperpara_indices = self._get_neighbor_model(self.hyperpara_indices_current)
			self.hyperpara_dict = self._get_hyperpara_dict_from_positions(self.hyperpara_indices)
			self.score, train_time, sklearn_model = self._get_score(self.hyperpara_dict)

			# Normalized score difference to have a factor for later use with temperature and random
			score_diff_norm = (self.score_current - self.score)/(self.score_current + self.score)
			
			if self.score > self.score_current:
				self.score_current = self.score
				self.hyperpara_indices_current = self.hyperpara_indices

				if self.score > self.score_best:
					self.score_best = self.score
					self.hyperpara_indices_best = self.hyperpara_indices

			elif np.exp( score_diff_norm/self.temp ) > rand:
				self.score_current = self.score
				self.hyperpara_indices_current = self.hyperpara_indices

		hyperpara_dict_best = self._get_hyperpara_dict_from_positions(self.hyperpara_indices_best)
		score_best, train_time, sklearn_model = self._get_score(hyperpara_dict_best)

		return sklearn_model, score_best, hyperpara_dict_best, train_time
		

