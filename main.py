from matplotlib import pyplot as plt
import numpy as np
import argparse

from event_Python import eventvision
from lib.spatio_temporal_feature import TimeSurface
from lib.utils import cosine_dist, euclidean_dist


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train digit recogniser')
    parser.add_argument('--input_files', action='store', nargs='+', default='datasets/mnist/Test/0/00004.bin',
                        help='Path to event file')

    args = parser.parse_args()

    ev = eventvision.read_dataset(args.input_files[0])

    # #############   plot time surface for whole event sequence  ###############

    # plot time context

    ts = TimeSurface(ev.height, ev.width, region_size=2, time_constant=10000 * 2)

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

    S_init = TimeSurface(ev.height, ev.width, region_size=2, time_constant=10000 * 2)

    # initialise and plot each of the time surface prototypes

    if False:
        for i in range(N_1):
            x = ev.data[i].x
            y = ev.data[i].y

            if ev.data[i].p:
                C_1_on[i][y, x] = 1
            else:
                C_1_off[i][y, x] = 1
    elif True:
        for i in range(N_1):
            x = ev.width / (N_1 + 1) * (i + 1)
            y = ev.height / (N_1 + 1) * (i + 1)

            C_1_on[i][y, x] = 1
            C_1_off[i][y, x] = 1
    else:
        for i in range(N_1):
            x = ev.data[i].x
            y = ev.data[i].y

            S_init.process_event(ev.data[i])

            if ev.data[i].p:
                C_1_on[i] = S_init.time_surface_on
            else:
                C_1_off[i] = S_init.time_surface_off

    fig, ax = plt.subplots(2, N_1, figsize=(25, 5))

    for i in range(N_1):
        ax[0, i].imshow(C_1_on[i])
        ax[1, i].imshow(C_1_off[i])
        ax[0, i].set_title('Time surface {}'.format(i))

    plt.show()

    # ############ Train time surface prototypes for layer 1 ############

    event_data = [eventvision.read_dataset(f).data for f in args.input_files]

    # initialise time surface
    S = TimeSurface(ev.height, ev.width, region_size=1, time_constant=10000 * 2)

    # TODO: should we have a p_on and p_off, to count separately for each prototype?
    p = [1] * N_1

    for e in ev.data: #[:20]:

        S.process_event(e)

        # plt.imshow(S.time_surface_on)
        # plt.show()

        # find closest cluster center (i.e. closest time surface prototype, according to euclidean distance)

        if e.p:
            dists = [euclidean_dist(c_k.reshape(-1), S.time_surface_on.reshape(-1)) for c_k in C_1_on]
        else:
            dists = [euclidean_dist(c_k.reshape(-1), S.time_surface_off.reshape(-1)) for c_k in C_1_off]

        k = np.argmin(dists)

        print('k:', k, dists)

        # update prototype that is closest to

        alpha = 0.01 / (1 + p[k] / 200.)

        if e.p:
            beta = cosine_dist(C_1_on[k].reshape(-1), S.time_surface_on.reshape(-1))
            C_1_on[k] += alpha * (S.time_surface_on - beta * C_1_on[k])
        else:
            beta = cosine_dist(C_1_off[k].reshape(-1), S.time_surface_off.reshape(-1))
            C_1_off[k] += alpha * (S.time_surface_off - beta * C_1_off[k])

        p[k] += 1

    print k
    print(e, dists, k, alpha, beta, p)

    fig, ax = plt.subplots(2, N_1, figsize=(25, 5))

    for i in range(N_1):
        ax[0, i].imshow(C_1_on[i])
        ax[1, i].imshow(C_1_off[i])
        ax[0, i].set_title('Time surface {}'.format(i))

    plt.show()
