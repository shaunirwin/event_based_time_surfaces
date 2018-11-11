from matplotlib import pyplot as plt
import numpy as np
import argparse
import os
import glob

from event_Python import eventvision
from lib.spatio_temporal_feature import TimeSurface
from lib.utils import cosine_dist, euclidean_dist
from lib.noise_filter import remove_isolated_pixels


def initialise_time_surface_prototypes(N, tau, r, width, height, events, init_method=3, plot=False):
    """
    Initialise time surface prototypes ("features" or "cluster centers") for a layer

    :param N: Number of features in this layer
    :param tau: time constant [us]
    :param r: radius [pixels]
    :param width: [pixels]
    :param height: [pixels]
    :param events: list of incoming events. Must be at least as many elements as N.
    :param init_method:
    :param plot: if True, plot the initialised time surfaces
    :return: N numpy arrays
    """

    C = [np.zeros((height, width)) for _ in range(N)]

    S = TimeSurface(height, width, region_size=r, time_constant=tau)

    # initialise each of the time surface prototypes

    if init_method == 1:
        for i in range(N):
            x = events[i].x
            y = events[i].y

            C[i][y, x] = 1
    elif init_method == 2:
        for i in range(N):
            x = width / (N + 1) * (i + 1)
            y = height / (N + 1) * (i + 1)

            C[i][y, x] = 1
    elif init_method == 2:
        # hard-coded locations
        C[0][12, 11] = 1
        C[1][13, 16] = 1
        C[2][26, 8] = 1
        C[3][24, 17] = 1
    else:
        for i in range(N):
            S.process_event(events[i])
            C[i] = S.time_surface

    # plot the initialised time surface prototypes

    if plot:
        fig, ax = plt.subplots(1, N, figsize=(20, 5))

        for i in range(N):
            ax[i].imshow(C[i])
            ax[i].set_title('Time surface {}'.format(i))

        plt.show()

    return C


def generate_layer_outputs(polarities, features, tau, r, width, height, events):
    """
    Generate events at the output of a layer from a stream of incoming events

    :param polarities: number of polarities of incoming events
    :param features: list of trained features for this layer
    :param tau: time constant [us]
    :param r: radius [pixels]
    :param width: [pixels]
    :param height: [pixels]
    :param events: list of incoming events. Must be at least as many elements as N.
    :return: events at output of layer
    """

    S = [TimeSurface(height, width, region_size=r, time_constant=tau)] * polarities
    events_out = []

    for e in events:
        # update time surface with incoming event. Select the time surface corresponding to polarity of event.

        S[e.p].process_event(e)

        # select the closest feature to this time surface

        dists = [euclidean_dist(feature.reshape(-1), S[e.p].time_surface.reshape(-1)) for feature in features]

        k = np.argmin(dists)

        # create output event

        e_out = e.copy()
        e_out.p = k
        events_out.append(e_out)

    return events_out


def main():
    parser = argparse.ArgumentParser(description='Train digit recogniser')
    parser.add_argument('--input_folders_training', action='store', nargs='+', default='datasets/mnist/Test/0',
                        help='Paths to folders containing event files')
    parser.add_argument('--num_files_per_folder', action='store', default=1, type=int,
                        help="Number of files to read in from each digit's folder")

    args = parser.parse_args()

    # get event data files within folders

    input_files_all = []

    for folder in args.input_folders_training:
        input_files = glob.glob(os.path.join(folder, '*.bin'))[:args.num_files_per_folder]
        input_files_all.extend(input_files)
        print('Num files from {}: {}'.format(folder, len(input_files)))

    ev = eventvision.read_dataset(input_files_all[0])

    # #############   plot time surface for whole event sequence  ###############

    # plot time context

    ts_1_1 = TimeSurface(ev.height, ev.width, region_size=2, time_constant=10000 * 2)
    ts_1_2 = TimeSurface(ev.height, ev.width, region_size=2, time_constant=10000 * 2)

    # set time to pause at
    t_pause = 70000

    # filter out outliers

    event_data = ev.data
    event_data_filt, _, _ = remove_isolated_pixels(event_data, eps=3, min_samples=20)

    for e in event_data_filt:
        if e.ts <= t_pause:
            if e.p:
                ts_1_1.process_event(e)
            else:
                ts_1_2.process_event(e)

    if True:
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

    N_1 = 4
    tau_1 = 20000
    r_1 = 2

    K_N = 2
    K_tau = 2
    K_r = 2

    N_2 = N_1 * K_N
    tau_2 = tau_1 * K_tau
    r_2 = r_1 * K_r

    C_1 = initialise_time_surface_prototypes(N_1, tau_1, r_1, ev.width, ev.height, event_data_filt, plot=False)

    # ############ Train time surface prototypes for layer 1 ############

    event_data = []
    event_data_filt = []

    for f in input_files_all:
        ev_data = eventvision.read_dataset(f).data
        ev_data_filt = remove_isolated_pixels(ev_data, eps=3, min_samples=20)[0]

        event_data.extend(ev_data)
        event_data_filt.extend(ev_data_filt)

    # initialise time surface
    S_on = TimeSurface(ev.height, ev.width, region_size=r_1, time_constant=tau_1)
    S_off = TimeSurface(ev.height, ev.width, region_size=r_1, time_constant=tau_1)

    p = [1] * N_1

    for e in event_data_filt:

        if e.p:
            S_on.process_event(e)
            S = S_on
        else:
            S_off.process_event(e)
            S = S_off

        # find closest cluster center (i.e. closest time surface prototype, according to euclidean distance)

        dists = [euclidean_dist(c_k.reshape(-1), S.time_surface.reshape(-1)) for c_k in C_1]

        k = np.argmin(dists)

        print('k:', k, dists)

        # update prototype that is closest to

        alpha = 0.01 / (1 + p[k] / 20000.)

        beta = cosine_dist(C_1[k].reshape(-1), S.time_surface.reshape(-1))
        C_1[k] += alpha * (S.time_surface - beta * C_1[k])

        p[k] += 1

    print k
    print(e, dists, k, alpha, beta, p)

    fig, ax = plt.subplots(1, N_1, figsize=(25, 5))

    for i in range(N_1):
        ax[i].imshow(C_1[i])
        ax[i].set_title('Layer 1. Time surface {} (p={})'.format(i, p[i]))

    plt.show()

    # ############ Train time surface prototypes for layer 2 ############

    # generate event data at output of layer 1 (using the trained features)

    event_data_2 = generate_layer_outputs(polarities=2, features=C_1, tau=tau_1, r=r_1, width=ev.width,
                                          height=ev.height, events=event_data_filt)

    # initialise and plot each of the time surface prototypes

    C_2 = initialise_time_surface_prototypes(N_2, tau_2, r_2, ev.width, ev.height, event_data_2, plot=True)

    # # initialise time surface
    # S_on = TimeSurface(ev.height, ev.width, region_size=r_2, time_constant=tau_2)
    # S_off = TimeSurface(ev.height, ev.width, region_size=r_2, time_constant=tau_2)
    #
    # p = [1] * N_1
    #
    # for e in event_data_filt:
    #
    #     if e.p:
    #         S_on.process_event(e)
    #         S = S_on
    #     else:
    #         S_off.process_event(e)
    #         S = S_off
    #
    #     # find closest cluster center (i.e. closest time surface prototype, according to euclidean distance)
    #
    #     dists = [euclidean_dist(c_k.reshape(-1), S.time_surface.reshape(-1)) for c_k in C_1]
    #
    #     k = np.argmin(dists)
    #
    #     print('k:', k, dists)
    #
    #     # update prototype that is closest to
    #
    #     alpha = 0.01 / (1 + p[k] / 20000.)
    #
    #     beta = cosine_dist(C_1[k].reshape(-1), S.time_surface.reshape(-1))
    #     C_1[k] += alpha * (S.time_surface - beta * C_1[k])
    #
    #     p[k] += 1
    #
    # print k
    # print(e, dists, k, alpha, beta, p)
    #
    # fig, ax = plt.subplots(1, N_1, figsize=(25, 5))
    #
    # for i in range(N_1):
    #     ax[i].imshow(C_1[i])
    #     ax[i].set_title('Layer 1. Time surface {} (p={})'.format(i, p[i]))
    #
    # plt.show()


if __name__ == '__main__':
    main()
