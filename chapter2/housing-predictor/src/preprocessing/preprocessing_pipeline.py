from preprocessing.attributes_adder import CombinedAttributesAdder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin

class CustomLabelBinarizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder = LabelBinarizer()

    def fit(self, x, y=0):
        self.encoder.fit(x)
        return self

    def transform(self, x, y=0):
        return self.encoder.transform(x)

num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler())
])

cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy="most_frequent")),
    ('label_binarizer', CustomLabelBinarizer())
])

preprocessing_pipeline = ColumnTransformer([
    ("num_transformer", num_transformer, ["longitude", "latitude", "housing_median_age",
     "total_rooms", "total_bedrooms", "population", "households", "median_income"]),
    ("cat_transformer", cat_transformer, ["ocean_proximity"])
])
