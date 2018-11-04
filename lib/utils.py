import numpy as np


def cosine_dist(x, y):
    """ Calculate cosine similarity between two matrices
    """

    return np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y)))
