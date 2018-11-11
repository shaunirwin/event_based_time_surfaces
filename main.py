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

    ts_1_1 = TimeSurface(ev.height, ev.width, region_size=2, time_constant=10000 * 2)
    ts_1_2 = TimeSurface(ev.height, ev.width, region_size=2, time_constant=10000 * 2)

    # set time to pause at
    t_pause = 70000

    for e in ev.data:
        if e.ts <= t_pause:
            if e.p:
                ts_1_1.process_event(e)
            else:
                ts_1_2.process_event(e)

    fig, ax = plt.subplots(2, 3, figsize=(10, 5))
    ax[0, 0].imshow(ts_1_1.latest_times)
    ax[0, 1].imshow(ts_1_1.time_context)
    ax[0, 2].imshow(ts_1_1.time_surface)
    ax[1, 0].imshow(ts_1_2.latest_times)
    ax[1, 1].imshow(ts_1_2.time_context)
    ax[1, 2].imshow(ts_1_2.time_surface)
    ax[0, 0].set_title('Latest times')
    ax[0, 1].set_title('Time context')
    ax[0, 2].set_title('Time surface')

    plt.show()

    # ############## Initialise time surface prototypes ##############

    # Choose number of prototypes for layer 1
    N_1 = 4
    tau_1 = 20000
    r_1 = 1

    C_1 = [np.zeros((ev.height, ev.width)) for _ in range(N_1)]

    S_init = TimeSurface(ev.height, ev.width, region_size=r_1, time_constant=tau_1)

    # initialise and plot each of the time surface prototypes

    if False:
        for i in range(N_1):
            x = ev.data[i].x
            y = ev.data[i].y

            C_1_on[i][y, x] = 1
    elif False:
        for i in range(N_1):
            x = ev.width / (N_1 + 1) * (i + 1)
            y = ev.height / (N_1 + 1) * (i + 1)

            C_1[i][y, x] = 1
    else:
        for i in range(N_1):
            x = ev.data[i].x
            y = ev.data[i].y

            S_init.process_event(ev.data[i])

            C_1[i] = S_init.time_surface

    fig, ax = plt.subplots(1, N_1, figsize=(25, 5))

    for i in range(N_1):
        ax[i].imshow(C_1[i])
        ax[i].set_title('Time surface {}'.format(i))

    plt.show()

    # ############ Train time surface prototypes for layer 1 ############

    event_data = []

    for f in args.input_files:
        event_data.extend(eventvision.read_dataset(f).data)

    # initialise time surface
    S_on = TimeSurface(ev.height, ev.width, region_size=r_1, time_constant=tau_1)
    S_off = TimeSurface(ev.height, ev.width, region_size=r_1, time_constant=tau_1)

    # TODO: should we have a p_on and p_off, to count separately for each prototype?
    p = [1] * N_1

    for e in event_data: #[:20]:

        if e.p:
            S_on.process_event(e)
            S = S_on
        else:
            S_off.process_event(e)
            S = S_off

        # plt.imshow(S.time_surface_on)
        # plt.show()

        # find closest cluster center (i.e. closest time surface prototype, according to euclidean distance)

        dists = [euclidean_dist(c_k.reshape(-1), S.time_surface.reshape(-1)) for c_k in C_1]

        k = np.argmin(dists)

        print('k:', k, dists)

        # update prototype that is closest to

        alpha = 0.01 / (1 + p[k] / 20000.)      # TODO: is this value set equal to the time constant or is it a coincidence?

        beta = cosine_dist(C_1[k].reshape(-1), S.time_surface.reshape(-1))
        C_1[k] += alpha * (S.time_surface - beta * C_1[k])

        p[k] += 1

    print k
    print(e, dists, k, alpha, beta, p)

    fig, ax = plt.subplots(1, N_1, figsize=(25, 5))

    for i in range(N_1):
        ax[i].imshow(C_1[i])
        ax[i].set_title('Time surface {}'.format(i))

    plt.show()
