import numpy as np
from biosiglive.interfaces.pytrigno_interface import PytrignoClient
from biosiglive.processing.data_processing import RealTimeProcessing
from biosiglive.gui.plot import LivePlot
from biosiglive.io.save_data import add_data_to_pickle, read_data
from time import sleep, time
try:
    import biorbd
except ImportError:
    pass


if __name__ == '__main__':
    try_offline = False
    if try_offline:
        # Get prerecorded data from pickle file for a shoulder abduction
        offline_emg = read_data("abd")["emg"]

    output_file_path = "trial_x"

    # init trigno community client
    trigno_interface = PytrignoClient(ip="localhost")

    # Add markerSet to Vicon interface
    n_electrodes = 2
    emg_processing = RealTimeProcessing()
    emg_processing.ma_win = 200
    emg_processing.emg_win = 2000
    emg_processing.emg_rate = 2000
    emg_processing.bpf_hcut = 435

    emg_plot = LivePlot()
    emg_plot.add_new_plot(plot_name="emg", channel_names=["muscle_1", "muscle_2"], plot_type="curve", nb_subplot=2)
    rplt, window, app, box = emg_plot.init_plot_window(plot=emg_plot.plot[0], use_checkbox=True)

    if not try_offline:
        trigno_interface.add_device(name="emg_test", range=(0, n_electrodes), type="emg", rate=2000, system_rate=100)

    time_to_sleep = 1/100

    offline_count = 0
    plot_wind = emg_processing.emg_win
    emg_to_plot = np.zeros((2, 20))
    raw_emg, emg_proc = [], []
    while True:
        tic = time()
        sample = trigno_interface.devices[-1].sample
        if try_offline:
            # Get prerecorded data
            emg_tmp = offline_emg[:2, offline_count:offline_count + sample]
            offline_count = 0 if offline_count > offline_emg.shape[1] - sample else offline_count + sample
        else:
            # Get last trigno frame and get emg data from it
            trigno_interface.get_frame()
            emg_tmp = trigno_interface.get_device_data()[0]

        raw_emg, emg_proc = emg_processing.process_emg(raw_emg,
                                                       emg_proc,
                                                       emg_tmp,
                                                       norm_emg=False,
                                                       )

        if emg_to_plot.shape[1] < plot_wind:
            emg_to_plot = np.append(emg_to_plot, emg_tmp, axis=1)
        else:
            emg_to_plot = np.append(emg_to_plot[:, sample:], emg_tmp, axis=1)
        emg_plot.update_plot_window(emg_plot.plot[0], emg_to_plot, app, rplt, box)

        # Save binary file
        add_data_to_pickle({"emg": emg_tmp}, output_file_path)

        loop_time = time() - tic
        real_time_to_sleep = time_to_sleep - loop_time
        if real_time_to_sleep > 0:
            sleep(real_time_to_sleep)
