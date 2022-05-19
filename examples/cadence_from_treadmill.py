import numpy as np
from biosiglive.interfaces.vicon_interface import ViconClient
from biosiglive.processing.data_processing import RealTimeProcessing
from biosiglive.gui.plot import LivePlot
from time import sleep, time


if __name__ == '__main__':
    show_cadence = True
    vicon_interface = ViconClient(init_now=False)
    vicon_interface.add_device("Treadmill", "generic_device", rate=1000, system_rate=100)
    vicon_interface.devices[-1].set_process_method(RealTimeProcessing().get_peaks)
    force_z, force_z_process = [], []
    if show_cadence:
        plot_app = LivePlot()
        plot_app.add_new_plot("cadence", "curve", ["force_z_R", "force_z_L", "force_z_R_raw", "force_z_L_raw"])
        rplt, window, app, box = plot_app.init_plot_window(plot=plot_app.plot[0], use_checkbox=True)
<<<<<<< Updated upstream
    nb_second = 10
=======
    nb_second = 5
>>>>>>> Stashed changes
    nb_min_frame = vicon_interface.devices[-1].rate * nb_second
    time_to_sleep = 1/vicon_interface.devices[-1].system_rate
    count = 0
    tic = time()
    cadence_wanted = 40

    time = np.linspace(0, np.pi*2 * cadence_wanted, 60000)
    amplitude = np.sin(time)
    F_r = [i if i > 0 else 0 for i in amplitude]
    F_l = [-i if i < 0 else 0 for i in amplitude]
    sample = 10
    c = 0
    force_z_tmp = np.zeros((2, sample))
    is_one = [False, False]
    while True:
<<<<<<< Updated upstream
        # data = vicon_interface.get_device_data(device_name="Treadmill")
        # force_z_tmp = data[0][[2, 8], :]
        force_z_tmp[0, :], force_z_tmp[1, :] = F_r[c:c + sample], F_l[c:c + sample]
        c = c + 10 if c + sample < len(F_r) else 0

        cadence, force_z_process, force_z, is_one = vicon_interface.devices[0].process_method(new_sample=force_z_tmp,
                                                                                      signal=force_z,
                                                                                      signal_proc=force_z_process,
                                                                                      threshold=0.01,
=======
        # data = vicon_interface.get_device_data(device_name="Treadmill", channel_names="Fz")
        # force_z_tmp = data[0][[2, 6], :]
        force_z_tmp = np.random.random((2, 10))
        cadence, force_z_process, force_z = vicon_interface.devices[0].process_method(new_sample=force_z_tmp,
                                                                                      signal=force_z,
                                                                                      signal_proc=force_z_process,
                                                                                      threshold=0.0005,
>>>>>>> Stashed changes
                                                                                      nb_min_frame=nb_min_frame,
                                                                                      is_one=is_one,
                                                                                      min_peaks_interval=50
                                                                                      )
        if show_cadence:
<<<<<<< Updated upstream
            plot_app.update_plot_window(plot_app.plot[0], np.concatenate((force_z_process, force_z), axis=0), app, rplt, box)

        if count == 1000:
=======
            plot_app.update_plot_window(plot_app.plot[0], force_z_process, app, rplt, box)

        if count == nb_min_frame / (vicon_interface.devices[-1].rate/vicon_interface.devices[-1].system_rate()):
>>>>>>> Stashed changes
            print(f"Mean cadence for the last {nb_second} s is :{cadence}")
            count = 0
        count += 1
        sleep(time_to_sleep)

