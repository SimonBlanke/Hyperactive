"""
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
"""

from sklearn.model_selection import cross_val_score


def add_dataset_name(self):
    return "Dataset_name", self.data_name


def get_number_of_instances(self):
    return "N_rows", int(self.X_train.shape[0])


def get_number_of_features(self):
    return "N_columns", int(self.X_train.shape[1])


def get_default_score(self):
    return (
        "cv_default_score",
        cross_val_score(self.model, self.X_train, self.y_train, cv=5).mean(),
    )
