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


import time
import datetime
import numpy as np
import pandas as pd
import multiprocessing

num_cores = multiprocessing.cpu_count()


def augment_dataset(X_train, y_train, n_drops=3, dataset_name=None):
	dataset_dict = {}
	dataset_dict['base'] = X_train

	dataset_dict_temp = dict(dataset_dict)
	dataset_dict['base'] = [X_train, y_train]

	#pool = multiprocessing.Pool(num_cores)
	#dataset_dict = zip(*pool.map(multiprocessing_helper, range(0, n_drops-2)))
	
	if n_drops > len(X_train.columns)-2:
		n_drops = len(X_train.columns)-2
		print('Number of drops to high for dataset. Setting n_drops to', n_drops)
	for i in range(n_drops):
		dataset_dict_temp = drop_feature(dataset_dict_temp)

		for key_temp in dataset_dict_temp:
			dataset_dict[key_temp] = [dataset_dict_temp[key_temp], y_train]

	return dataset_dict

'''
def multiprocessing_helper(n_drops):
	dataset_dict_temp = drop_feature(dataset_dict_temp)

	for key_temp in dataset_dict_temp:
		dataset_dict[key_temp] = [dataset_dict_temp[key_temp], y_train]

	return dataset_dict
'''
def drop_feature(dataset_dict):
	dataset_dict_temp = {}
	already_dropped = []

	for dataset_key in dataset_dict:
		dataset = dataset_dict[dataset_key]
		features = dataset.columns

		for feature in features:
			dataset_dropped = dataset.drop(feature, axis=1)
			append = True
			for dropped_features in already_dropped:
				if dropped_features == list(dataset_dropped.columns):
					append = False

			already_dropped.append(list(dataset_dropped.columns))
			if append == True:
				key = dataset_key+'_'+str(feature)
				dataset_dict_temp[key] = dataset_dropped
		
	return dataset_dict_temp


def drop_instance(dataset_dict):
	dataset_dict_temp = {}
	already_dropped = []

	for dataset_key in dataset_dict:
		dataset = dataset_dict[dataset_key][0]
		rows = dataset.index

		for row in rows:
			dataset_dropped = dataset.drop(row, axis=0)
			append = True
			for dropped_rows in already_dropped:
				if dropped_rows == list(dataset_dropped.index):
					append = False

			already_dropped.append(list(dataset_dropped.index))
			if append == True:
				key = dataset_key+'_'+str(row)
				dataset_dict_temp[key] = dataset_dropped
		
	return dataset_dict_temp










