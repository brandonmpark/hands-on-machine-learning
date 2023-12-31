from scipy.ndimage import shift
import numpy as np


def shift_image(image, dx, dy):
    image = image.reshape((28, 28))
    shifted_image = shift(image, [dy, dx])
    return shifted_image.reshape([-1])

def augment(X, y):
    X_augmented = [image for image in X]
    y_augmented = [label for label in y]

    for image, label in zip(X, y):
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            X_augmented.append(shift_image(image, dx, dy))
            y_augmented.append(label)

    return np.array(X_augmented), np.array(y_augmented)