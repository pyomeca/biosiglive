import numpy as np
from biosiglive.interfaces.vicon_interface import ViconClient
from biosiglive.processing.data_processing import RealTimeProcessing
from biosiglive.gui.plot import LivePlot
from time import sleep, time


if __name__ == '__main__':
    show_cadence = True
    vicon_interface = ViconClient(init_now=True)
    vicon_interface.add_device("Treadmill", "generic_device", rate=2000, system_rate=100)
    vicon_interface.devices[-1].set_process_method(RealTimeProcessing().get_peaks)
    force_z, force_z_process = [], []
    if show_cadence:
        plot_app = LivePlot()
        plot_app.add_new_plot("cadence", "curve", ["force_z_R", "force_z_L", "force_z_R_raw", "force_z_L_raw"])
        rplt, window, app, box = plot_app.init_plot_window(plot=plot_app.plot[0], use_checkbox=True)

    nb_second = 60
    nb_min_frame = vicon_interface.devices[-1].rate * nb_second
    time_to_sleep = 1/vicon_interface.devices[-1].system_rate
    count = 0
    tic = time()

    # cadence_wanted = 80
    # time = np.linspace(0, np.pi*2 * cadence_wanted/2, 120000)
    # amplitude = np.sin(time)
    # F_r = [i if i > 0 else -i for i in amplitude]
    # F_l = [-i if i < 0 else 0 for i in amplitude]
    # import matplotlib.pyplot as plt
    # plt.plot(F_l)
    # plt.plot(F_r)
    # plt.show()
    sample = 20
    c = 0
    force_z_tmp = np.zeros((2, sample))
    is_one = [False, False]
    while True:
        vicon_interface.get_frame()
        data = vicon_interface.get_device_data(device_name="Treadmill")
        force_z_tmp = data[0][[2, 8], :]
        # force_z_tmp[0, :], force_z_tmp[1, :] = F_r[c:c + sample], F_l[c:c + sample]
        # plt.plot(force_z_tmp[0, :])
        # plt.plot(force_z_tmp[1, :])
        # plt.show()
        # c = c + sample if c + sample < len(F_r) else 0
        cadence, force_z_process, force_z, is_one = vicon_interface.devices[0].process_method(new_sample=force_z_tmp,
                                                                                      signal=force_z,
                                                                                      signal_proc=force_z_process,
                                                                                      threshold=0.01,
                                                                                      nb_min_frame=nb_min_frame,
                                                                                      is_one=is_one,
                                                                                      min_peaks_interval=2000
                                                                                      )
        if show_cadence:
            plot_app.update_plot_window(plot_app.plot[0], np.concatenate((force_z_process, force_z), axis=0), app, rplt, box)

        if count == 100:
            print(f"Mean cadence for the last {nb_second} s is :{cadence}")
            count = 0
        count += 1
        sleep(time_to_sleep)

