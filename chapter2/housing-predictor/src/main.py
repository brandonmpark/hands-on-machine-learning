from sklearn.model_selection import train_test_split
from fetch_data import fetch_housing_data, load_housing_data
from evaluate import evaluate
from joblib import dump

fetch_housing_data()
housing = load_housing_data()

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
housing = train_set.drop("median_house_value", axis=1)
housing_labels = train_set["median_house_value"].copy()

# from modeling.linear_regression import linear_regression_grid
# linear_regression_grid.fit(housing, housing_labels)
# print(linear_regression_grid.best_estimator_)
# evaluate(linear_regression_grid.best_estimator_, housing, housing_labels)

# from modeling.svr import svr_grid
# svr_grid.fit(housing, housing_labels)
# print(svr_grid.best_estimator_)
# evaluate(svr_grid.best_estimator_, housing, housing_labels)

# from modeling.random_forest import random_forest_grid
# random_forest_grid.fit(housing, housing_labels)
# print(random_forest_grid.best_estimator_)
# evaluate(random_forest_grid.best_estimator_, housing, housing_labels)

from modeling.random_forest_grid import random_forest_grid
random_forest_grid.fit(housing, housing_labels)
print(random_forest_grid.best_estimator_)
evaluate(random_forest_grid.best_estimator_, housing, housing_labels)

final_model = random_forest_grid.best_estimator_
test_housing = test_set.drop("median_house_value", axis=1)
test_housing_labels = test_set["median_house_value"].copy()
evaluate(final_model, test_housing, test_housing_labels)
dump(final_model, '../models/final_model.joblib')
