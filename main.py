# import sys
from matplotlib import pyplot as plt
import numpy as np

# sys.path.append('event_Python')

from event_Python import eventvision
from lib.spatio_temporal_feature import Feature


if __name__ == '__main__':
    ev = eventvision.read_dataset(r'datasets\mnist\Test\0\00004.bin')

    # #############   plot time surface (ON events only) for whole event sequence  ##############+

    # plot time context

    feat = Feature(ev.height, ev.width, region_size=1, time_constant=10000 * 2)

    # set time to pause at
    t_pause = 70000

    for e in ev.data:
        if e.ts <= t_pause:
            if e.p:
                feat.process_event(e)

    fig, ax = plt.subplots(1, 3, figsize=(10, 5))
    ax[0].imshow(feat.latest_times_on)
    ax[1].imshow(feat.time_context_on)
    ax[2].imshow(feat.time_surface_on)
    ax[0].set_title('Latest times')
    ax[1].set_title('Time context')
    ax[2].set_title('Time surface')

    plt.show()

    # ############## Initialise time surface prototypes ##############

    # Choose number of prototypes for layer 1
    N_1 = 4

    C_1 = [Feature(ev.height, ev.width, region_size=1, time_constant=10000 * 2) for _ in range(N_1)]

    # initialise and plot each of the time surface prototypes

    fig, ax = plt.subplots(1, N_1, figsize=(25, 5))

    for i in range(N_1):
        C_1[i].process_event(ev.data[i])

        ax[i].imshow(C_1[i].time_surface_on)
        ax[i].imshow(C_1[i].time_surface_on)
        ax[i].set_title('Time surface {}'.format(i))

    plt.show()

    # ############ Train time surface prototypes for layer 1 ############

    # initialise time surface
    S = Feature(ev.height, ev.width, region_size=1, time_constant=10000 * 2)

    p = [1] * N_1

    for e in ev.data:  # [:88]:

        S.process_event(e)

        # find closest cluster center (i.e. closest time surface prototype, according to euclidean distance)

        dists = [np.linalg.norm(c_k.time_surface_on - S.time_surface_on) for c_k in C_1]

        k = np.argmin(dists)

        # update prototype that is closest to

        alpha = 0.01 / (1 + p[k] / 2000.)
        beta = np.dot(C_1[k].time_surface_on, S.time_surface_on) / (
                    np.linalg.norm(C_1[k].time_surface_on) * np.linalg.norm(S.time_surface_on))

        #     C_1[k].time_surface += alpha * (S.time_surface - beta * C_1[k].time_surface)

        p[k] += 1

    print k
    print(e, dists, k, alpha, beta, p)

    fig, ax = plt.subplots(1, N_1, figsize=(25, 5))

    for i in range(N_1):
        C_1[i].process_event(ev.data[i])

        ax[i].imshow(C_1[i].time_surface)
        ax[i].set_title('Time surface {}'.format(i))

    plt.show()
