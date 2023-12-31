from sklearn.svm import SVR
from preprocessing.preprocessing_pipeline import preprocessing_pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

param_grid = {
    'preprocessing_pipeline__num_transformer__imputer__strategy': ['mean', 'median', 'most_frequent'],
    'preprocessing_pipeline__num_transformer__attribs_adder__add_rooms_per_household': [True, False],
    'preprocessing_pipeline__num_transformer__attribs_adder__add_population_per_household': [True, False],
    'preprocessing_pipeline__num_transformer__attribs_adder__add_bedrooms_per_room': [True, False],
    'preprocessing_pipeline__cat_transformer__imputer__strategy': ['most_frequent', 'constant'],
    'model__kernel': ['linear', 'rbf'],
    'model__C': [0.1, 1, 10, 100],
    'model__gamma': ['scale', 'auto'],
}

pipeline = Pipeline([
    ('preprocessing_pipeline', preprocessing_pipeline),
    ('model', SVR())
])

svr_grid = RandomizedSearchCV(
    pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', refit=True, n_jobs=-1)
