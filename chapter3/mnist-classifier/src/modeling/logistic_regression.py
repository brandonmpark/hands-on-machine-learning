from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from preprocessing.preprocessing_pipeline import preprocessing_pipeline

param_grid = {
    'model__penalty': ['l1', 'l2'],
    'model__C': [0.1, 1, 10, 100],
    'model__solver': ['liblinear', 'saga'],
}

pipeline = Pipeline([
    ('preprocessing_pipeline', preprocessing_pipeline),
    ('model', LogisticRegression())
])

logistic_regression_grid = RandomizedSearchCV(
    pipeline, param_grid, cv=3, scoring='accuracy', refit=True, n_jobs=-1, verbose=2)
