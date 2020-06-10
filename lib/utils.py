import numpy as np


def cosine_dist(x, y):
    """
    Calculate cosine similarity between two vectors
    """

    return np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y)))


def euclidean_dist(x, y):
    """
    Calculate euclidean distance between two vectors
    """

    return np.sqrt(np.sum((x - y) ** 2))


def euclidean_norm_dist(x, y):
    """
    Calculate euclidean distance, normalised by number of "features"
    """

    x = x.astype(float)
    y = y.astype(float)

    return np.sqrt(np.sum((x / x.sum() - y / y.sum()) ** 2))


def bhattacharyya_dist(x, y):
    """
    Calculate Bhattacharyya distance, as explained in Bhattacharyya 1943 "On a measure of divergence between two
    statistical populations defined by probability distributions".

    This measure determines the relative closeness of two distributions, for example to measure separability between two
    classes for classification. It is a generalisation of the Mahalanobis distance, for cases where the standard
    deviation of the two classes are different.
    """

    x = x.astype(float)
    y = y.astype(float)

    return -np.log(np.sum(np.sqrt(np.multiply(x / x.sum(), y / y.sum()))))
