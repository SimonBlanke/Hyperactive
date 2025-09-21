from sklearn.svm import SVC
from hyperactive.integrations.sklearn import OptCV
from hyperactive.opt import GridSearchSk as GridSearch

param_grid = {"kernel": ["linear", "rbf"], "C": [1, 10]}
tuned_svc = OptCV(SVC(), GridSearch(param_grid))

# 2. fitting the tuned estimator:
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

tuned_svc.fit(X_train, y_train)

# 3. obtaining predictions
y_pred = tuned_svc.predict(X_test)

# 4. obtaining best parameters and best estimator
best_params = tuned_svc.best_params_
best_estimator = tuned_svc.best_estimator_
