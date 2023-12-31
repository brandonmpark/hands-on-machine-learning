from sklearn.neighbors import KNeighborsClassifier
from preprocessing.preprocessing_pipeline import preprocessing_pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

param_grid = {
    'model__n_neighbors': [3, 5, 7, 9],
    'model__weights': ['uniform', 'distance'],
    # 'model__p': [1, 2],
    'model__p': [2],
}

pipeline = Pipeline([
    ('preprocessing_pipeline', preprocessing_pipeline),
    ('model', KNeighborsClassifier())
])

knn_grid = RandomizedSearchCV(pipeline, param_grid, cv=3,
                        scoring='accuracy', refit=True, n_jobs=-1, verbose=2)