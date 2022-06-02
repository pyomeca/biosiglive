import numpy as np
from biosiglive.interfaces.vicon_interface import ViconClient
from biosiglive.io.save_data import add_data_to_pickle, read_data
from time import sleep, time
try:
    import biorbd
except ImportError:
    pass


if __name__ == '__main__':
    try_offline = True
    init_now = False if try_offline else True
    if try_offline:
        # Get prerecorded data from pickle file for a shoulder abduction
        offline_markers = read_data("abd")["markers"][:3, :, :]

    output_file_path = "trial_x"

    #init Vicon Client
    vicon_interface = ViconClient(init_now=init_now)

    # Add markerSet to Vicon interface
    vicon_interface.add_markers(rate=100, unlabeled=False, subject_name="subject_1")

    time_to_sleep = 1/vicon_interface.markers[0].rate
    offline_count = 0
    while True:
        tic = time()

        if try_offline:
            # Get prerecorded data
            markers_tmp = offline_markers[:, :, offline_count][:, :, np.newaxis]
            offline_count = 0  if offline_count == offline_markers.shape[2] - 1 else offline_count + 1

        else:
            # Get last vicon frame and get markers data from it
            vicon_interface.get_frame()
            markers_tmp = vicon_interface.get_markers_data()[0]

        # Save binary file
        add_data_to_pickle({"markers": markers_tmp}, output_file_path)

        loop_time = time() - tic
        real_time_to_sleep = time_to_sleep - loop_time
        if real_time_to_sleep > 0:
            sleep(real_time_to_sleep)
