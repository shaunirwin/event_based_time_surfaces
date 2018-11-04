import numpy as np

from lib.utils import cosine_dist


def test_cosine_dist_both_vectors_same_direction():
    a = np.array([1, 2, 3])

    b = a.copy() * 2

    assert cosine_dist(a, b) == 1


def test_cosine_dist_if_vectors_orthogonal():
    a = np.array([0, 1])

    b = np.array([1, 0])

    assert cosine_dist(a, b) == 0


def test_cosine_dist_both_vectors_opposite_direction():
    a = np.array([1, 2, 3])

    b = - a.copy() * 2

    assert cosine_dist(a, b) == -1
