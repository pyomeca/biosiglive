"""
This example shows how to disseminate EMG data using biosiglive and a dedicated interface.
First an interface is created, then a device is added to the interface. For EMG data, two interfaces are available:
ViconClient to stream data from a vicon Nexus software via the Vicon Data Stream SDK (https://www.vicon.com/software/datastream-sdk/). The Pytrigno interface for streaming data from the Delsys trigno community SDK (https://delsys.com/sdk/).
If you want to try this example offline, you can use the provided custom interface named 'MyInterface' which you can use as a standard interface.

The EMG device taking as input argument :
- the number of electrodes
- the type of the device (the allowed types are listed in the DeviceType class)
- the name of the device (for the Vicon system, it must be the same name as the device to stream).
- the flow of the device.
- the process method to use. Here, the process_emg method is used.
- any other argument needed for the processing method. (you can see the possible arguments in the documentation of the process_emg method).

If you want to display the data in real time, you can use the biosiglive plot classes, please take a look at the live_plot.py example.
After initializing the interface and the device, the data exchange is done in a loop.
The data must first be received from the source by the get_device_data method.
After that, the data can be used as is or processed using the process() method of the Device class.
In this function you can pass a method if you want to use a different method than the default one and every needed argument for that function as well.
"""

from biosiglive.enums import DeviceType
from custom_interface import MyInterface
from biosiglive.gui.plot import LivePlot
from biosiglive import (
    ViconClient,
    PytrignoClient,
    save,
    RealTimeProcessingMethod,
    PlotType,
)
from time import sleep, time


if __name__ == "__main__":
    try_offline = False

    output_file_path = "trial_x.bio"
    if try_offline:
        interface = MyInterface(system_rate=100, data_path="abd.bio")
    else:
        interface = PytrignoClient(ip="127.0.0.1")  # or interface = ViconClient(ip="localhost", system_rate=100)

    n_electrodes = 2
    muscle_names = [
        "Pectoralis major",
        "Deltoid anterior",
        # "Deltoid medial",
        # "Deltoid posterior",
    ]
    interface.add_device(
        nb_channels=n_electrodes,
        device_type=DeviceType.Emg,
        name="emg",
        rate=2000,
        device_data_file_key="emg",
        processing_method=RealTimeProcessingMethod.ProcessEmg,
        moving_average=True,
    )

    emg_plot = LivePlot(
        name="emg", rate=100, plot_type=PlotType.Curve, nb_subplots=n_electrodes, channel_names=muscle_names
    )
    emg_plot.init(plot_windows=500, y_labels="Processed EMG (mV)")

    emg_raw_plot = LivePlot(
        name="emg_raw", rate=100, plot_type=PlotType.Curve, nb_subplots=n_electrodes, channel_names=muscle_names
    )
    emg_raw_plot.init(plot_windows=10000, colors=(255, 0, 0), y_labels="EMG (mV)")

    time_to_sleep = 1 / 100
    count = 0
    file_path = "data_test.bio"
    while True:
        tic = time()
        raw_emg = interface.get_device_data(device_name="emg")
        emg_proc = interface.devices[0].process()
        emg_plot.update(emg_proc[:, -1:])
        emg_raw_plot.update(raw_emg)
        count += 1
        # save({'emg_raw': raw_emg, 'proc_emg': emg_proc}, file_path, add_data=True)
        # if try_offline:
        #     loop_time = time() - tic
        #     real_time_to_sleep = time_to_sleep - loop_time
        #     if real_time_to_sleep > 0:
        #         sleep(real_time_to_sleep)
