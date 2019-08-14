# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from sklearn.model_selection import cross_val_score


# def add_dataset_name(self):
#     return "Dataset_name", self.data_name


def get_number_of_instances(self):
    return "N_rows", int(self.X_train.shape[0])


def get_number_of_features(self):
    return "N_columns", int(self.X_train.shape[1])


def get_default_score(self):
    return (
        "cv_default_score",
        cross_val_score(self.model, self.X_train, self.y_train, cv=3).mean(),
    )
