"""
This example shows how to use the biosiglive library to get the cadence from a treadmill.
First an interface is created (here Vicon) and then a device (here a Treadmill) is added to the interface.
The Device class take several arguments here are the most important ones:
    - nb_channels: int
        Number of channels of the device.
    - name: str
        Name of the device.
    - rate: int
        Rate of the device.
    - processing_method: RealTimeProcessingMethods
        Method to process the data when calling the *.process() method. Here getpeak method is used.
    - any other argument needed by the processing method.
       here the getpeak method take the following arguments:
       threshold: float
              Threshold to detect a peak.
        min_peak_interval: int
                Minimum interval between two peaks. Expressed in number of frame.
Note that a custom interface is also available from the example 'custom_interface.py' and it allows the user
to run the examples without any device connection by streaming data from a provided data file.
If you want to try this example offline you can use the custom interface provided named "MyInterface" and you can use
 it as a standard interface.

If you want to display the data in real time you can use the biosiglive plot classes. You tan take a look at the
live_plot.py example to see how to use the LivePlot class.

Once everything initialized the data streaming take place in a loop.
The data are retrieved from the interface using the get_device_data method with the device name in argument.
After that the data might be used as this or can be processed using the process method of the device.
Device data are stored in a buffer inside the device class so no data need to be sent to the process method.
If needed the raw data are available in the device.raw_data buffer. The length of the buffer is defined by the
data buffer size argument of the device class. The default value is the rate of the device. When the buffer is full
the oldest data are replaced discarded and the new ones are added.

If the display is needed the plot.update() method is called with the data to display as argument.
the example run till the user stop it.
"""

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
        data_buffer_size=2000,
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
