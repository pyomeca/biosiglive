import numpy as np
from biosiglive.interfaces.vicon_interface import ViconClient
from biosiglive.processing.data_processing import RealTimeProcessing
from biosiglive.gui.plot import LivePlot
from time import sleep, time


if __name__ == '__main__':
    show_cadence = False
    vicon_interface = ViconClient(init_now=True)
    vicon_interface.add_device("Treadmill", "generic_device", rate=2000, system_rate=100)
    vicon_interface.devices[-1].set_process_method(RealTimeProcessing().get_peaks)
    force_z, force_z_process = [], []
    if show_cadence:
        plot_app = LivePlot()
        plot_app.add_new_plot("cadence", "curve", ["force_z_R", "force_z_L", "force_z_R_raw", "force_z_L_raw"])
        rplt, window, app, box = plot_app.init_plot_window(plot=plot_app.plot[0], use_checkbox=True)

    nb_second = 20
    print_every = 10  # seconds
    nb_min_frame = vicon_interface.devices[-1].rate * nb_second
    time_to_sleep = 1 / vicon_interface.devices[-1].system_rate
    count = 0
    c = 0
    is_one = [False, False]
    while True:
        tic = time()
        vicon_interface.get_frame()
        data = vicon_interface.get_device_data(device_name="Treadmill")
        force_z_tmp = data[0][[2, 8], :]
        cadence, force_z_process, force_z, is_one = vicon_interface.devices[0].process_method(new_sample=force_z_tmp,
                                                                                      signal=force_z,
                                                                                      signal_proc=force_z_process,
                                                                                      threshold=0.2,
                                                                                      nb_min_frame=nb_min_frame,
                                                                                      is_one=is_one,
                                                                                      min_peaks_interval=1300
                                                                                      )
        if show_cadence:
            plot_app.update_plot_window(plot_app.plot[0], np.concatenate((force_z_process, force_z), axis=0), app, rplt, box)
        # print(force_z)
        if count == print_every * vicon_interface.devices[-1].system_rate:
            print(f"Mean cadence for the last {nb_second} s is :{cadence}")
            count = 0
        count += 1
        # loop_time = time() - tic
        # real_time_to_sleep = time_to_sleep - loop_time
        # print(vicon_interface.get_frame_number())
        # if real_time_to_sleep > 0:
        #     sleep(time_to_sleep - loop_time)
        # else:
