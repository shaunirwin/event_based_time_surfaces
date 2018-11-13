import numpy as np
from copy import deepcopy
from sklearn.cluster import DBSCAN


def remove_isolated_pixels(event_data, eps=3, min_samples=20):
    """
    Removes isolated events from the event stream.

    These are events that do not have nearby neighbours in time and space

    :param event_data:
    :param eps: DB SCAN parameter
    :param min_samples: DB SCAN parameter
    :return: filtered event data
    """

    points = np.array([(e.x, e.y, e.ts) for e in event_data])

    # normalise time axis to similar magnitude to spatial axes

    x_dist = points[:, 0].max() - points[:, 0].min()
    t_dist = points[:, 2].max() - points[:, 2].min()
    t_scale = t_dist / x_dist
    points[:, 2] /= t_scale

    # filter out outliers

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)

    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    points_filt = points[core_samples_mask, :]

    event_data_copy = deepcopy(event_data)

    event_data_filt = [e for i, e in enumerate(event_data_copy) if core_samples_mask[i]]

    return event_data_filt, points, points_filt
