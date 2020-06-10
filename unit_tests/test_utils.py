import numpy as np

from lib.utils import cosine_dist, euclidean_dist, euclidean_norm_dist, bhattacharyya_dist


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


def test_euclidean_distance_for_same_vectors_gives_0():
    a = np.array([1, 2])

    b = np.array([1, 2])

    assert euclidean_dist(a, b) == 0


def test_euclidean_distance_for_different_vectors_gives_non_zero_scalar():
    a = np.array([1, 0])

    b = np.array([0, 1])

    assert euclidean_dist(a, b) == np.sqrt(2)


def test_normalised_euclidean_distance_for_same_vectors_gives_0():
    a = np.array([2, 0])

    b = np.array([1, 0])

    assert euclidean_norm_dist(a, b) == 0


def test_normalised_euclidean_distance_for_different_vectors_gives_gt_0():
    a = np.array([2, 0])

    b = np.array([0, 1])

    assert euclidean_norm_dist(a, b) == np.sqrt(2)


def test_bhattacharyya_distance_for_same_vectors_gives_0():
    a = np.array([1, 2, 3])

    b = np.array([1, 2, 3])

    assert bhattacharyya_dist(a, b) == 0


def test_bhattacharyya_distance_for_different_vectors_gives_gt_0():
    a = np.array([1, 2, 3])

    b = np.array([1, 2, 4])

    assert bhattacharyya_dist(a, b) > 0
