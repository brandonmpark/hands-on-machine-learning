from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from preprocessing.preprocessing_pipeline import preprocessing_pipeline

param_grid = {
    'preprocessing_pipeline__num_transformer__imputer__strategy': ['mean', 'median', 'most_frequent'],
    'preprocessing_pipeline__cat_transformer__imputer__strategy': ['most_frequent', 'constant'],
    'model__n_estimators': [3, 10, 30],
    'model__max_features': [2, 4, 6, 8],
    'model__bootstrap': [False],
}

pipeline = Pipeline([
    ('preprocessing_pipeline', preprocessing_pipeline),
    ('model', RandomForestRegressor())
])

random_forest_grid = GridSearchCV(
    pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', refit=True, n_jobs=-1)
