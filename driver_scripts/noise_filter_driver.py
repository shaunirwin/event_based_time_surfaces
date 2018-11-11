import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from event_Python import eventvision
from lib.noise_filter import remove_isolated_pixels


if __name__ == '__main__':
    f = 'datasets/mnist/Test/8/00062.bin'
    ev = eventvision.read_dataset(f)

    # extract 3D points (x-y-time)

    ev_filt, points, points_filt = remove_isolated_pixels(ev.data, eps=3, min_samples=20)

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.scatter3D(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap='Greens')

    # plt.show()

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.scatter3D(points_filt[:, 0], points_filt[:, 1], points_filt[:, 2], c=points_filt[:, 2], cmap='Greens')

    plt.show()
