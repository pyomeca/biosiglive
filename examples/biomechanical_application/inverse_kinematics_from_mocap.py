import numpy as np
from biosiglive.interfaces.vicon_interface import ViconClient
from biosiglive.processing.data_processing import RealTimeProcessing
from biosiglive.processing.msk_functions import kalman_func
from biosiglive.io.save_data import add_data_to_pickle, read_data
from biosiglive.gui.plot import LivePlot
from time import sleep, time

try:
    import biorbd
except ImportError:
    pass


def load_offline_markers(trial):
    mat = read_data(trial)
    markers = mat["markers"][:3, :, :]
    return markers


if __name__ == "__main__":
    try_offline = False
    if try_offline:
        offline_trial = "abd"
        offline_markers = load_offline_markers(offline_trial)
        init_now = False
    else:
        init_now = True

    show_skeleton = False
    output_file_path = "trial_x"
    plot_fequency = 100
    model_path = "model/Wu_Shoulder_Model_mod_wt_wrapp.bioMod"
    vicon_interface = ViconClient(init_now=init_now)
    vicon_interface.add_markers(rate=100, unlabeled=False, subject_name="Clara")

    # model = biorbd.Model(model_path)

    if show_skeleton:
        skeleton_plot = LivePlot()
        skeleton_plot.msk_model = model_path
        skeleton_plot.add_new_plot(plot_type="skeleton")
        skeleton_plot.set_skeleton_plot_options(show_floor=True)
        skeleton_plot.init_plot_window(skeleton_plot.plot[0])

    # q_est = np.zeros((model.nbQ(), 100))
    time_to_sleep = 1 / vicon_interface.markers[0].rate
    ratio = int(vicon_interface.markers[0].rate / plot_fequency)
    plot_count = 1
    offline_count = 0
    kalman = None
    while True:
        tic = time()
        if try_offline:
            markers_tmp = offline_markers[:, :, offline_count][:, :, np.newaxis]
            if offline_count == offline_markers.shape[2] - 1:
                offline_count = 0
            else:
                offline_count += 1
        else:
            vicon_interface.get_frame()
            markers_tmp = vicon_interface.get_markers_data()[0]

        # states = vicon_interface.get_kinematics_from_markers(markers_tmp, model, return_qdot=False)
        if show_skeleton:
            if plot_count == ratio:
                skeleton_plot.update_plot_window(skeleton_plot.plot[0], states[:, -1])
                plot_count = 1
            else:
                plot_count += 1
        print(markers_tmp)
        add_data_to_pickle({"markers": markers_tmp}, output_file_path)

        loop_time = time() - tic
        if show_skeleton:
            print("plot_frequency: ", 1 / loop_time)
        real_time_to_sleep = time_to_sleep - loop_time
        if real_time_to_sleep > 0:
            sleep(real_time_to_sleep)
        # else:
        #     print("Warning: the loop took too long to run")
