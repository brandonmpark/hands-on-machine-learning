from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
import numpy as np
from evaluate import evaluate
from joblib import dump
from preprocessing.augment import augment

mnist = fetch_openml('mnist_784')
X, y = np.array(mnist["data"]), np.array(mnist["target"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, y_train = augment(X_train, y_train)

from modeling.knn import knn_grid
knn_grid.fit(X_train, y_train)
evaluate(knn_grid.best_estimator_, X_test, y_test)

dump(knn_grid.best_estimator_, 'models/knn_grid.joblib')

# from modeling.random_forest import random_forest_grid
# random_forest_grid.fit(X_train, y_train)
# print(random_forest_grid.best_estimator_)
# evaluate(random_forest_grid.best_estimator_, X_train, y_train)

# from modeling.svc import svc_grid
# svc_grid.fit(X_train, y_train)
# print(svc_grid.best_estimator_)
# evaluate(svc_grid.best_estimator_(), X_train, y_train)

# from modeling.sgd_classifier import sgd_classifier_grid
# sgd_classifier_grid.fit(X_train, y_train)
# print(sgd_classifier_grid.best_estimator_)
# evaluate(sgd_classifier_grid.best_estimator_, X_train, y_train)

# from modeling.logistic_regression import logistic_regression_grid
# logistic_regression_grid.fit(X_train, y_train)
# print(logistic_regression_grid.best_estimator_)
# evaluate(logistic_regression_grid.best_estimator_, X_train, y_train)