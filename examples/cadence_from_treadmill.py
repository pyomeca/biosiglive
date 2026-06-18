"""
This example shows how to use the biosiglive library to get the cadence of a treadmill.
First, an interface is created (here Vicon) and then a device (here a treadmill) is added to the interface. If you want to try this example offline, you can use the provided custom interface named "MyInterface" and you can use it as a standard interface.
The data can be plotted in real time at each loop, please see the live_plot.py example.

"""

from biosiglive.interfaces.vicon_interface import ViconClient
from time import time, sleep
from custom_interface import MyInterface
import numpy as np
from biosiglive import RealTimeProcessingMethod, InterfaceType, DeviceType, LivePlot, PlotType


if __name__ == "__main__":
    interface = None
    plot_curve = LivePlot(
        name="curve",
        plot_type=PlotType.Curve,
        nb_subplots=4,
        channel_names=["1", "2", "3", "4"],
    )
    # plot_curve = LivePlot(
    #     name="strike",
    #     plot_type=PlotType.Curve,
    #     nb_subplots=2,
    #     channel_names=["1", "2"],
    # )
    plot_curve.init(plot_windows=10000, y_labels=["Strikes", "Strikes", "Force (N)", "Force (N)"])
    interface_type = InterfaceType.Custom
    if interface_type == InterfaceType.Custom:
        interface = MyInterface(system_rate=100, data_path="walk.bio")
    elif interface_type == InterfaceType.ViconClient:
        interface = ViconClient(system_rate=100)
    nb_second = 10
    interface.add_device(
        9,
        name="Treadmill",
        device_type=DeviceType.Generic,
        rate=1000,
        processing_method=RealTimeProcessingMethod.GetPeaks,
        data_buffer_size=1000 * nb_second,
        processing_window=1000 * nb_second,
        device_data_file_key="treadmill",
        threshold=0.2,
        min_peaks_interval=800,
    )
    print_every = 10  # seconds
    count = 0
    tic_bis = time()
    while True:
        tic = time()
        data = interface.get_device_data(device_name="Treadmill")
        force_z_tmp = data[[2, 8], :]
        peaks, force_z_process = interface.get_device("Treadmill").process()
        if peaks:
            cadence = peaks[2] + peaks[8]
        else:
            cadence = 0
        if count == print_every * interface.devices[-1].system_rate:
            print(f"Loop time: {time() - tic_bis}")
            print(f"Mean cadence for the last {nb_second} s is :{cadence}")
            tic_bis = time()
            count = 0
        count += 1
        plot_curve.update(np.append(force_z_process[[2, 8], -10:], force_z_tmp, axis=0))
        loop_time = time() - tic

        # if interface_type == InterfaceType.Custom:
        #     if (1/100) - loop_time > 0:
        #         sleep((1/100) - loop_time - 0.002)
