# Cadence from treadmill

This example shows how to use the biosiglive library to get the cadence of a treadmill.
First, an interface is created (here Vicon) and then a device (here a treadmill) is added to the interface. If you want to try this example offline, you can use the provided custom interface named "MyInterface" and you can use it as a standard interface.
The number of second is used to define on which windows the cadence will be computed. For instance here 30 second means that the computed cadence is the number of step for 30 seconds. As the data are collected at high frequency consider to use a reduced windows.

```
from biosiglive.interfaces.vicon_interface import ViconClient
from time import time, sleep
from custom_interface import MyInterface
from biosiglive import RealTimeProcessingMethod, InterfaceType, DeviceType


if __name__ == "__main__":
    interface = None
    interface_type = InterfaceType.Custom
    if interface_type == InterfaceType.Custom:
        interface = MyInterface(system_rate=100, data_path="walk.bio")
    elif interface_type == InterfaceType.ViconClient:
        interface = ViconClient(system_rate=100)
    nb_second = 30
    interface.add_device(
        9,
        name="Treadmill",
        device_type=DeviceType.Generic,
        rate=1000,
        processing_method=RealTimeProcessingMethod.GetPeaks,
        data_buffer_size=1000*nb_second,
        processing_window=1000*nb_second,
        device_data_file_key="treadmill",
        threshold=0.6,
        min_peaks_interval=1300,
    )
    print_every = 10  # seconds
    count = 0
    tic_bis = time()
    while True:
        data = interface.get_device_data(device_name="Treadmill")
        force_z_tmp = data[[2, 8], :]
        cadence, force_z_process = interface.get_device("Treadmill").process()
        if count == print_every * interface.devices[-1].system_rate:
            print(f"Mean cadence for the last {nb_second} s is :{cadence}")
            tic_bis = time()
            count = 0
        count += 1
        if interface_type == InterfaceType.Custom:
            sleep(1/100)

```
