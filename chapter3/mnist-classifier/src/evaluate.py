import numpy as np
from sklearn.metrics import accuracy_score


def evaluate(model, X, y):
    predictions = model.predict(X)
    print("Accuracy: ", accuracy_score(y, predictions))
