from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_rooms_per_household=True, add_population_per_household=True, add_bedrooms_per_room=True):
        self.add_rooms_per_household = add_rooms_per_household
        self.add_population_per_household = add_population_per_household
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
        if self.add_rooms_per_household:
            X = np.c_[X, rooms_per_household]
        if self.add_population_per_household:
            X = np.c_[X, population_per_household]
        if self.add_bedrooms_per_room:
            X = np.c_[X, bedrooms_per_room]
        return X
