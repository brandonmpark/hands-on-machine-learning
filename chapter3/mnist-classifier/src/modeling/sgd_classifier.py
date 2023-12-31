from sklearn.linear_model import SGDClassifier
from preprocessing.preprocessing_pipeline import preprocessing_pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

param_grid = {
    'model__loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
    'model__penalty': ['l2', 'l1', 'elasticnet'],
    'model__alpha': [0.0001, 0.001, 0.01, 0.1],
    'model__learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
    'model__eta0': [0.01, 0.1, 1],
    'model__power_t': [0.1, 0.5, 1],
    'model__early_stopping': [True, False],
    'model__validation_fraction': [0.1, 0.2, 0.3],
    'model__n_iter_no_change': [5, 10, 15],
    'model__average': [True, False],
}

pipeline = Pipeline([
    ('preprocessing_pipeline', preprocessing_pipeline),
    ('model', SGDClassifier())
])

sgd_classifier_grid = RandomizedSearchCV(
    pipeline, param_grid, cv=3, scoring='accuracy', refit=True, n_jobs=-1, verbose=2)
