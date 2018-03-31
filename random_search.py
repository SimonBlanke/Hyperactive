import os
import sys
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing


from sklearn import datasets


from dict1 import classifier_config_dict
from dict1 import classifier_config_dict1
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from importlib import import_module
from sklearn.metrics import f1_score

from sklearn.model_selection import cross_val_score

from functools import partial

iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


num_cores = multiprocessing.cpu_count()
print(num_cores, 'CPU threads available')


def search(N_searches, X_train, y_train, scoring, ML_dict=classifier_config_dict1, cv=5):

	model = None
	hyperpara_values = []
	hyperpara_names = []

	hyperpara_value = []
	hyperpara_name = []
	hyperpara_dict = {}

	dict1_len = len(ML_dict)
	#print('dict1_len = ', dict1_len)
	rand1 = random.randint(0, dict1_len-1)
	#print('rand = ', rand1)

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

		'''
		for k, value in zip(range(dict3_len), dict1[model][key_hyperpara]):

			print(key_hyperpara, ': ', value, '\n')

			if k == rand2:
				hyperpara_value.append(value)
				
		'''
	sklearn, submod_func = model.rsplit('.', 1)
	module = import_module(sklearn)
	model = getattr(module, submod_func)

	ML_model = model(**hyperpara_dict)

	#print(ML_model)
	#ML_model.fit(X_train, y_train)

	scores = cross_val_score(ML_model, X, y, scoring=scoring, cv=cv)
	score = scores.mean()

	#y_pred = ML_model.predict(X_test)
	#score = f1_score(y_test, y_pred, average='micro')

	return ML_model, score




def random_search(ML_dict, X_train, y_train, scoring, N_pipelines=100, T_search_time=None, cv=5):

	N_best_models = 1

	pool = multiprocessing.Pool(num_cores)
	search1 = partial(search, ML_dict=ML_dict, X_train=X_train, y_train=y_train, scoring=scoring, cv=cv)
	models, scores = zip(*pool.map(search1, range(0, N_pipelines)))

	scores = np.array(scores)
	index_best_scores = scores.argsort()[-N_best_models:][::-1]

	best_score = scores[index_best_scores]
	best_pipeline = models[index_best_scores[0]]

	'''
	for i in range(len(index_best_scores)):
		print(models[index_best_scores[i]], '\n')
	'''

	return best_pipeline, best_score




def apply_random_search(ML_dict, X_train, y_train, scoring, N_pipelines=None, T_search_time=None, cv=5):
	if T_search_time is not None and N_pipelines is None:
		start = time.time()
		random_search(ML_dict=ML_dict, X_train=X_train, y_train=y_train, scoring=scoring, N_pipelines=1000, T_search_time=None, cv=cv)
		search_time = time.time() - start

		N_pipelines = int(1000 * T_search_time/search_time)
		print(N_pipelines)


		start = time.time()
		best_pipeline, best_score = random_search(ML_dict=ML_dict, X_train=X_train, y_train=y_train, scoring=scoring, N_pipelines=N_pipelines, T_search_time=None, cv=cv)
		print('search_time: ', time.time() - start)


	print('best_score: ', best_score, '\n')
	print('best_pipeline: ', best_pipeline, '\n')





#apply_random_search(ML_dict=classifier_config_dict1, X_train=X, y_train=y, scoring='f1_micro', T_search_time=30, cv=3)





