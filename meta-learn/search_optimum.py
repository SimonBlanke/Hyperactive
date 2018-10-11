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
import pickle
import numpy as np
import pandas as pd

from functools import partial
from scipy.optimize import minimize

def search_optimum(self, X_train, y_train):
	features_from_dataset = self._get_features_from_dataset(X_train, y_train)

	
	function_part = partial(self._function, features_from_dataset=features_from_dataset)


	x0 = None
	best_hyperpara = minimize(function_part, x0, method='nelder-mead', options={'xtol': 1e-8, 'disp': True})



def _function(self, features_from_dataset, features_from_model):
	x = pd.concat([features_from_dataset, features_from_model], axis=1)
	return -self.meta_regressor.predict(x)

