from matplotlib import pyplot as plt
import numpy as np

from event_Python import eventvision
from lib.spatio_temporal_feature import TimeSurface
from lib.utils import cosine_dist


if __name__ == '__main__':
    ev = eventvision.read_dataset(r'datasets\mnist\Test\0\00004.bin')

    # #############   plot time surface for whole event sequence  ###############

    # plot time context

    ts = TimeSurface(ev.height, ev.width, region_size=1, time_constant=10000 * 2)

    # set time to pause at
    t_pause = 70000

    for e in ev.data:
        if e.ts <= t_pause:
            ts.process_event(e)

    fig, ax = plt.subplots(2, 3, figsize=(10, 5))
    ax[0, 0].imshow(ts.latest_times_on)
    ax[0, 1].imshow(ts.time_context_on)
    ax[0, 2].imshow(ts.time_surface_on)
    ax[1, 0].imshow(ts.latest_times_off)
    ax[1, 1].imshow(ts.time_context_off)
    ax[1, 2].imshow(ts.time_surface_off)
    ax[0, 0].set_title('Latest times')
    ax[0, 1].set_title('Time context')
    ax[0, 2].set_title('Time surface')

    plt.show()

    # ############## Initialise time surface prototypes ##############

    # Choose number of prototypes for layer 1
    N_1 = 4

    C_1_on = [np.zeros((ev.height, ev.width)) for _ in range(N_1)]
    C_1_off = [np.zeros((ev.height, ev.width)) for _ in range(N_1)]

    # initialise and plot each of the time surface prototypes

    for i in range(N_1):
        x = ev.data[i].x
        y = ev.data[i].y

        if ev.data[i].p:
            C_1_on[i][y, x] = 1
        else:
            C_1_off[i][y, x] = 1

    fig, ax = plt.subplots(2, N_1, figsize=(25, 5))

    for i in range(N_1):
        ax[0, i].imshow(C_1_on[i])
        ax[1, i].imshow(C_1_off[i])
        ax[0, i].set_title('Time surface {}'.format(i))

    plt.show()

    # ############ Train time surface prototypes for layer 1 ############

    # initialise time surface
    S = TimeSurface(ev.height, ev.width, region_size=1, time_constant=10000 * 2)

    p = [1] * N_1

    for e in ev.data:  # [:88]:

        S.process_event(e)

        # find closest cluster center (i.e. closest time surface prototype, according to euclidean distance)

        dists = [cosine_dist(c_k.reshape(-1), S.time_surface_on.reshape(-1)) for c_k in C_1_on]

        k = np.argmin(dists)

        # update prototype that is closest to

        alpha = 0.01 / (1 + p[k] / 2000.)
        beta = cosine_dist(C_1_on[k].reshape(-1), S.time_surface_on.reshape(-1))

        C_1_on[k] += alpha * (S.time_surface_on - beta * C_1_on[k])

        p[k] += 1

    print k
    print(e, dists, k, alpha, beta, p)

    fig, ax = plt.subplots(2, N_1, figsize=(25, 5))

    for i in range(N_1):
        ax[0, i].imshow(C_1_on[i])
        ax[1, i].imshow(C_1_off[i])
        ax[0, i].set_title('Time surface {}'.format(i))

    plt.show()
