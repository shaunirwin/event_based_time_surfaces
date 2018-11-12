from matplotlib import pyplot as plt
import numpy as np
import argparse
import os
import glob

from event_Python import eventvision
from lib.spatio_temporal_feature import TimeSurface
from lib.utils import cosine_dist, euclidean_dist
from lib.noise_filter import remove_isolated_pixels


def visualise_time_surface_for_event_stream(N, tau, r, width, height, events):
    # plot time context

    ts_1_1 = TimeSurface(height, width, region_size=r, time_constant=tau)
    ts_1_2 = TimeSurface(height, width, region_size=r, time_constant=tau)

    # set time to pause at
    t_pause = 70000

    # filter out outliers

    event_data = events
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


def train_layer(C, N, tau, r, width, height, events, polarities, layer_number, plot=False):
    """
    Train the time surface prototypes in a single layer

    :param C: time surface prototypes (i.e. cluster centers) that have been initialised
    :param N:
    :param tau:
    :param r:
    :param width:
    :param height:
    :param events:
    :param polarities:
    :param layer_number: the number of this layer
    :param plot:
    :return:
    """

    # initialise time surface
    S = [TimeSurface(height, width, region_size=r, time_constant=tau)] * polarities

    p = [1] * N

    for e in events:
        S[e.p].process_event(e)

        # find closest cluster center (i.e. closest time surface prototype, according to euclidean distance)

        dists = [euclidean_dist(c_k.reshape(-1), S[e.p].time_surface.reshape(-1)) for c_k in C]

        k = np.argmin(dists)

        print('k:', k, dists)

        # update prototype that is closest to

        alpha = 0.01 / (1 + p[k] / 20000.)

        beta = cosine_dist(C[k].reshape(-1), S[e.p].time_surface.reshape(-1))
        C[k] += alpha * (S[e.p].time_surface - beta * C[k])

        p[k] += 1

    print k
    print(e, dists, k, alpha, beta, p)

    if plot:
        fig, ax = plt.subplots(1, N, figsize=(25, 5))

        for i in range(N):
            ax[i].imshow(C[i])
            ax[i].set_title('Layer {}. Time surface {} (p={})'.format(layer_number, i, p[i]))

        plt.show()


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

    # --------------- configure network parameters ---------------

    N_1 = 4
    tau_1 = 20000
    r_1 = 2

    K_N = 2
    K_tau = 2
    K_r = 2

    N_2 = N_1 * K_N
    tau_2 = tau_1 * K_tau
    r_2 = r_1 * K_r

    N_3 = N_2 * K_N
    tau_3 = tau_2 * K_tau
    r_3 = r_2 * K_r

    # --------------- get event data files within folders ---------------

    input_files_all = []

    for folder in args.input_folders_training:
        input_files = glob.glob(os.path.join(folder, '*.bin'))[:args.num_files_per_folder]
        input_files_all.extend(input_files)
        print('Num files from {}: {}'.format(folder, len(input_files)))

    ev = eventvision.read_dataset(input_files_all[0])

    # filter out outliers

    event_data = []
    event_data_filt = []

    for f in input_files_all:
        ev_data = eventvision.read_dataset(f).data
        ev_data_filt = remove_isolated_pixels(ev_data, eps=3, min_samples=20)[0]

        event_data.extend(ev_data)
        event_data_filt.extend(ev_data_filt)

    # plot time surface for single event sequence

    visualise_time_surface_for_event_stream(N_1, tau_1, r_1, ev.width, ev.height, ev.data)

    # --------------- Train time surface prototypes for layer 1 ---------------

    C_1 = initialise_time_surface_prototypes(N_1, tau_1, r_1, ev.width, ev.height, event_data_filt, plot=True)

    train_layer(C_1, N_1, tau_1, r_1, ev.width, ev.height, event_data_filt, polarities=2, layer_number=1, plot=True)

    # --------------- Train time surface prototypes for layer 2 ---------------

    # generate event data at output of layer 1 (using the trained features)

    event_data_2 = generate_layer_outputs(polarities=2, features=C_1, tau=tau_1, r=r_1, width=ev.width,
                                          height=ev.height, events=event_data_filt)

    C_2 = initialise_time_surface_prototypes(N_2, tau_2, r_2, ev.width, ev.height, event_data_2, plot=True)

    train_layer(C_2, N_2, tau_2, r_2, ev.width, ev.height, event_data_2, polarities=N_1, layer_number=2, plot=True)

    # --------------- Train time surface prototypes for layer 2 ---------------

    # generate event data at output of layer 1 (using the trained features)

    event_data_3 = generate_layer_outputs(polarities=N_1, features=C_2, tau=tau_2, r=r_2, width=ev.width,
                                          height=ev.height, events=event_data_2)

    C_3 = initialise_time_surface_prototypes(N_3, tau_3, r_3, ev.width, ev.height, event_data_3, plot=True)

    train_layer(C_3, N_3, tau_3, r_3, ev.width, ev.height, event_data_3, polarities=N_2, layer_number=3, plot=True)


if __name__ == '__main__':
    main()
