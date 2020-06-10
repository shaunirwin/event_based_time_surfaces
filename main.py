from __future__ import print_function
import argparse
import os
import glob
from random import shuffle

from event_Python import eventvision
from lib.noise_filter import remove_isolated_pixels
from lib.layer_operations import visualise_time_surface_for_event_stream, initialise_time_surface_prototypes, \
    generate_layer_outputs, train_layer


def main():
    parser = argparse.ArgumentParser(description='Train digit recogniser')
    parser.add_argument('--input_folders_training', action='store', nargs='+', default='datasets/mnist/Test/0',
                        help='Paths to folders containing event files')
    parser.add_argument('--num_files_per_folder', action='store', default=1, type=int,
                        help="Number of files to read in from each digit's folder")

    args = parser.parse_args()

    # --------------- configure network parameters ---------------

    N_1 = 4
    tau_1 = 20000.
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

    # shuffle order of files to prevent reading same digit's files consecutively

    shuffle(input_files_all)

    ev = eventvision.read_dataset(input_files_all[0])

    # filter out outliers

    event_data = []
    event_data_filt = []

    for f in input_files_all:
        ev_data = eventvision.read_dataset(f).data
        ev_data_filt = remove_isolated_pixels(ev_data, eps=3, min_samples=20)[0]

        # ensure time stamps in event stream are monotonically increasing

        if len(event_data) > 0:
            ts_start_0 = event_data[-1].ts

            for i in range(len(ev_data)):
                ev_data[i].ts += ts_start_0

        if len(event_data_filt) > 0:
            ts_start_1 = event_data_filt[-1].ts

            for i in range(len(ev_data_filt)):
                ev_data_filt[i].ts += ts_start_1

        event_data.extend(ev_data)
        event_data_filt.extend(ev_data_filt)

        print('length event stream:', len(ev_data), len(ev_data_filt))

    # plot time surface for single event sequence

    visualise_time_surface_for_event_stream(N_1, tau_1, r_1, ev.width, ev.height, ev.data)

    # --------------- Train time surface prototypes for layer 1 ---------------

    C_1 = initialise_time_surface_prototypes(N_1, tau_1, r_1, ev.width, ev.height, event_data_filt, plot=True)

    train_layer(C_1, N_1, tau_1, r_1, ev.width, ev.height, event_data_filt, num_polarities=2, layer_number=1, plot=True)

    # --------------- Train time surface prototypes for layer 2 ---------------

    # generate event data at output of layer 1 (using the trained features)

    event_data_2 = generate_layer_outputs(num_polarities=2, features=C_1, tau=tau_1, r=r_1, width=ev.width,
                                          height=ev.height, events=event_data_filt)

    C_2 = initialise_time_surface_prototypes(N_2, tau_2, r_2, ev.width, ev.height, event_data_2, plot=True)

    train_layer(C_2, N_2, tau_2, r_2, ev.width, ev.height, event_data_2, num_polarities=N_1, layer_number=2, plot=True)

    # --------------- Train time surface prototypes for layer 2 ---------------

    # generate event data at output of layer 1 (using the trained features)

    event_data_3 = generate_layer_outputs(num_polarities=N_1, features=C_2, tau=tau_2, r=r_2, width=ev.width,
                                          height=ev.height, events=event_data_2)

    C_3 = initialise_time_surface_prototypes(N_3, tau_3, r_3, ev.width, ev.height, event_data_3, plot=True)

    train_layer(C_3, N_3, tau_3, r_3, ev.width, ev.height, event_data_3, num_polarities=N_2, layer_number=3, plot=True)


if __name__ == '__main__':
    main()
