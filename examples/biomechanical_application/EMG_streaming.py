import numpy as np
from custom_interface import MyInterface
from biosiglive import (
LivePlot,
add_data_to_pickle,
read_data,
ViconClient,
RealTimeProcessingMethod,
RealTimeProcessing,
)

from time import sleep, time
try:
    import biorbd
except ImportError:
    pass


def get_custom_function(device_interface):
    custom_processing = RealTimeProcessing(data_rate=device_interface.get_device(name="emg").rate, processing_windows=1000)
    custom_processing.bpf_lcut = 10
    custom_processing.bpf_hcut = 425
    custom_processing.lpf_lcut = 5
    custom_processing.lp_butter_order = 4
    custom_processing.bp_butter_order = 2
    custom_processing.moving_average_windows = 200
    return custom_processing.process_emg


if __name__ == '__main__':
    try_offline = True

    output_file_path = "trial_x.bio"
    if try_offline:
        interface = MyInterface(system_rate=100)
    else:
        # init trigno community client
        interface = ViconClient(ip="localhost", system_rate=100)

    # Add markerSet to Vicon interface
    n_electrodes = 1

    # Add device to Vicon interface
    interface.add_device(nb_channels=n_electrodes, device_type="emg", name="emg", rate=2000)

    # Add plot
    emg_plot = LivePlot(name="emg", channel_names=["raw_data", "custom_emg_process", "proc_emg"], plot_type="curve", nb_subplots=3)
    emg_plot.init(use_checkbox=True, plot_windows=[interface.devices[0].rate, interface.devices[0].rate, interface.system_rate])

    time_to_sleep = 1/100

    offline_count = 0

    while True:
        tic = time()
        emg_tmp = interface.get_device_data(device_name="emg")

        emg_proc = interface.devices[0].process(method=RealTimeProcessingMethod.ProcessEmg, moving_average_window=200)
        emg_proc_custom = interface.devices[0].process(method=RealTimeProcessingMethod.Custom, custom_function=get_custom_function, low_pass_filter=True, moving_average=False)
        emg_plot.update([emg_tmp, emg_proc_custom[:, -20:], emg_proc[:, -1:]])


        # Save binary file
        add_data_to_pickle({"emg": emg_tmp}, output_file_path)

        loop_time = time() - tic
        real_time_to_sleep = time_to_sleep - loop_time
        if real_time_to_sleep > 0:
            sleep(real_time_to_sleep)
