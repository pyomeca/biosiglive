import numpy as np
from biosiglive.interfaces.vicon_interface import ViconClient
from biosiglive.processing.data_processing import RealTimeProcessing
from biosiglive.gui.plot import LivePlot
from time import sleep, time


if __name__ == '__main__':
    show_cadence = False
    vicon_interface = ViconClient()
    vicon_interface.add_device("Treadmill", "generic_device", rate=1000, system_rate=100)
    vicon_interface.devices[-1].set_process_method(RealTimeProcessing().get_peaks)
    force_z, force_z_process = [], []
    if show_cadence:
        plot_app = LivePlot()
        plot_app.add_new_plot("cadence", "curve", ["force_z_R", "force_z_L"])
        rplt, window, app, box = plot_app.init_plot_window(plot=plot_app.plot[0], use_checkbox=True)
    nb_second = 1
    nb_min_frame = vicon_interface.devices[-1].rate * nb_second
    time_to_sleep = 1/vicon_interface.devices[-1].system_rate
    count = 0
    tic = time()
    while True:
        data = vicon_interface.get_device_data(device_name="Treadmill", channel_names="Fz")
        force_z_tmp = data[0][[2, 6], :]
        cadence, force_z_process, force_z = vicon_interface.devices[0].process_method(new_sample=force_z_tmp,
                                                                                      signal=force_z,
                                                                                      signal_proc=force_z_process,
                                                                                      threshold=5,
                                                                                      nb_min_frame=nb_min_frame,
                                                                                      )
        if show_cadence:
            plot_app.update_plot_window(plot_app.plot[0], force_z, app, rplt, box)

        if count == nb_min_frame / (vicon_interface.devices[-1].rate/vicon_interface.devices[-1].system_rate):
            print(f"Mean cadence for the last {nb_second} s is :{cadence}")
            count = 0
        count += 1
        sleep(time_to_sleep)

