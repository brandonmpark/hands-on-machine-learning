from sklearn.ensemble import RandomForestClassifier
from preprocessing.preprocessing_pipeline import preprocessing_pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

param_grid = {
    'model__n_estimators': [100, 200, 300, 400],
    'model__max_depth': [5, 10, 15, 20, 25],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 5],
    'model__max_features': ['sqrt', 'log2'],
    'model__bootstrap': [True, False]
}

pipeline = Pipeline([
    ('preprocessing_pipeline', preprocessing_pipeline),
    ('model', RandomForestClassifier())
])

random_forest_grid = RandomizedSearchCV(pipeline, param_grid, cv=3,
                              scoring='accuracy', refit=True, n_jobs=-1, verbose=2)
