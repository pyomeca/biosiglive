import numpy as np
from biosiglive.interfaces.vicon_interface import ViconClient
from time import time
from custom_interface import MyInterface
from biosiglive import LivePlot, PlotType, RealTimeProcessingMethod, InterfaceType


if __name__ == "__main__":
    show_plot = False
    interface = None
    plot = []
    interface_type = InterfaceType.Custom
    if interface_type == InterfaceType.Custom:
        interface = MyInterface(system_rate=100, data_path="walk.bio")
    elif interface_type == InterfaceType.ViconClient:
        interface = ViconClient(system_rate=100)

    interface.add_device(
        9,
        name="Treadmill",
        device_type="generic_device",
        rate=2000,
        processing_method=RealTimeProcessingMethod.GetPeaks,
        threshold=0.2,
        min_peaks_interval=1300,
    )
    if show_plot:
        plot = LivePlot(
            name="strikes",
            rate=100,
            plot_type=PlotType.Curve,
            nb_subplots=4,
            channel_names=["Rigth strike", "Left strike", "Rigth force", "Left force"],
        )
        plot.init(plot_windows=1000, y_labels=["Strikes", "Strikes", "Force (N)", "Force (N)"])

    nb_second = 20
    print_every = 10  # seconds
    nb_min_frame = interface.devices[-1].rate * nb_second
    count = 0
    while True:
        tic = time()
        data = interface.get_device_data(device_name="Treadmill")
        force_z_tmp = data[[2, 8], :]
        cadence, force_z_process = interface.get_device("Treadmill").process()
        if show_plot:
            plot.update(np.concatenate((force_z_process, force_z_tmp), axis=0))
        if count == print_every * interface.devices[-1].system_rate:
            print(f"Mean cadence for the last {nb_second} s is :{cadence}")
            count = 0
        count += 1
        if interface_type == InterfaceType.Custom:
            loop_time = time() - tic
            real_time_to_sleep = (1 / interface.devices[-1].system_rate) - loop_time
