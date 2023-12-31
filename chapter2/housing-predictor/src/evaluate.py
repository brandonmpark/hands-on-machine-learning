from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import numpy as np

def evaluate(model, X, y):
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    rmse = np.sqrt(mse)
    print("RMSE: ", rmse)
    scores = cross_val_score(
        model, X, y, scoring="neg_mean_squared_error", cv=5)
    rmse_scores = np.sqrt(-scores)
    print("Scores: ", rmse_scores)
    print("Mean: ", rmse_scores.mean())
    print("Standard deviation: ", rmse_scores.std())
