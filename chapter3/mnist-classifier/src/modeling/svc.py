from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from preprocessing.preprocessing_pipeline import preprocessing_pipeline

param_grid = {
    'model__C': [0.1, 1, 10, 100],
    'model__gamma': [0.01, 0.1, 1, 10],
    'model__kernel': ['linear', 'rbf'],
}

pipeline = Pipeline([
    ('preprocessing_pipeline', preprocessing_pipeline),
    ('model', SVC())
])

svc_grid = RandomizedSearchCV(
    pipeline, param_grid, cv=3, scoring='accuracy', refit=True, n_jobs=-1, verbose=2)
