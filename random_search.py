import os
import sys
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing

from importlib import import_module
from functools import partial
from sklearn.model_selection import cross_val_score


num_cores = multiprocessing.cpu_count()
print(num_cores, 'CPU threads available')


def random_search(N_searches, X_train, y_train, scoring, ML_dict, cv):

	random.seed(N_searches)

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




def multiprocessing_helper_func(ML_dict, X_train, y_train, scoring, cv, N_pipelines=100, T_search_time=None):

	N_best_models = 1

	pool = multiprocessing.Pool(num_cores)
	random_search_obj = partial(random_search, ML_dict=ML_dict, X_train=X_train, y_train=y_train, scoring=scoring, cv=cv)
	models, scores = zip(*pool.map(random_search_obj, range(0, N_pipelines)))

	scores = np.array(scores)
	index_best_scores = scores.argsort()[-N_best_models:][::-1]

	best_score = scores[index_best_scores]
	best_pipeline = models[index_best_scores[0]]

	return best_pipeline, best_score




def apply_random_search(ML_dict, X_train, y_train, scoring, N_pipelines=None, T_search_time=None, cv=5):
	if T_search_time is not None and N_pipelines is None:
		start = time.time()
		multiprocessing_helper_func(ML_dict=ML_dict, X_train=X_train, y_train=y_train, scoring=scoring, N_pipelines=1000, T_search_time=None, cv=cv)
		search_time = time.time() - start

		N_pipelines = int(1000 * T_search_time/search_time)

		start = time.time()
		best_pipeline, best_score = multiprocessing_helper_func(ML_dict=ML_dict, X_train=X_train, y_train=y_train, scoring=scoring, N_pipelines=N_pipelines, T_search_time=None, cv=cv)
		print('search_time: ', time.time() - start)
		print('Number of searched pipelines: ', N_pipelines, '\n')

	elif T_search_time is None and N_pipelines is not None:

		start = time.time()
		best_pipeline, best_score = multiprocessing_helper_func(ML_dict=ML_dict, X_train=X_train, y_train=y_train, scoring=scoring, N_pipelines=N_pipelines, T_search_time=None, cv=cv)
		print('search_time: ', time.time() - start)
		print('Number of searched pipelines: ', N_pipelines, '\n')


	print('best_score: ', best_score, '\n')
	print('best_pipeline: ', best_pipeline, '\n')








