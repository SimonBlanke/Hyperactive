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



from .base import BaseOptimizer


class ParticleSwarm_Optimizer(BaseOptimizer):

	def __init__(self, ml_search_dict, n_searches, scoring, n_particles=3, n_jobs=1, cv=5, verbosity=0):
		super().__init__(ml_search_dict, n_searches, scoring, n_jobs, cv, verbosity)

		self._search = self._start_particle_swarm_optimization

		self.ml_search_dict = ml_search_dict
		self.n_searches = n_searches
		self.scoring = scoring
		
		self.n_particles = n_particles

		self.n_jobs = n_jobs
		self.cv = cv
		self.verbosity = verbosity


		self.score = {}
		self.hyperpara_indices = {}
		self.hyperpara_dict = {}
		self.model_str = {}

		self.score_best = {}
		self.hyperpara_indices_best = {}
		self.hyperpara_dict_best = {}
		self.model_str_best = {}

		self.score_current = {}		
		self.hyperpara_indices_current = {}
		self.hyperpara_dict_current = {}		
		self.model_str_current = {}

		self.score_best_global = 0	
		self.hyperpara_indices_best_global = 0
		self.hyperpara_dict_best_global = 0		
		self.model_str_best_global = 0


	def _initialize_positions(self):
		for i in range(self.n_particles):
			self.model_str[i], self.hyperpara_dict[i], self.hyperpara_indices[i] = self._get_random_position()
			self.score[i], train_time = self._get_score(self.model_str[i], self.hyperpara_dict[i])

			self.score_best[i] = self.score[i]
			self.hyperpara_indices_best[i] = self.hyperpara_indices[i]
			self.hyperpara_dict_best[i] = self.hyperpara_dict[i]
			self.model_str_best[i] = self.model_str[i]

		self._find_best_particle()


	def _find_best_particle(self):
		best_score = 0
		best_particle = None

		for particle in self.score_best:
			score = self.score_best[particle]
			if score > best_score:
				best_score = score
				best_particle = particle

		self.score_best_global = best_score
		self.hyperpara_dict_best_global = self.hyperpara_dict_best[particle]
		self.hyperpara_indices_best_global = self.hyperpara_indices_best[particle]


	def _start_particle_swarm_optimization(self, n_searches):
		n_steps = int(self.n_searches/self.n_jobs)
		print('n_steps ', n_steps)

		self._initialize_positions()

		'''
		for i in range(n_steps):
			

			for particle in range(self.n_particles):
				print(particle)

		'''

		print('\n\n\n')
		return 0, 0, 0, 0


















