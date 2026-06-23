from examples.custom_interface import MyInterface
from biosiglive import RealTimeProcessingMethod, InterfaceType, DeviceType, LivePlot, PlotType
from biosiglive.interfaces.qualisys_interface import QualisysClient
# Import Stimulator class
from pyScienceMode import Channel, Device
from pyScienceMode import RehastimP24 as St
import qtm_rt
import numpy as np
import multiprocessing as mp
from time import time, sleep
import asyncio
import logging
import xml.etree.ElementTree as ET
from collections import deque

async def stim():
    interface = None
    plot_curve = LivePlot(
        name="curve",
        plot_type=PlotType.Curve,
        nb_subplots=2,
        channel_names=["1", "2"],
    )

    plot_curve.init(plot_windows=1000, y_labels=["Strikes", "Force (N)"])
    interface_type = InterfaceType.QualisysClient
    if interface_type == InterfaceType.Custom:
        interface = MyInterface(system_rate=100, data_path="walk.bio")
    elif interface_type == InterfaceType.QualisysClient:
        interface = await QualisysClient.create(ip="192.168.254.1", system_rate=100, port=22224)

    nb_second = 10
    await interface.add_device(
        nb_channels=12,
        device_type="force_plate",
        name="force_plate",
        data_buffer_size=100,
        rate=1000,
        device_data_file_key="force_plate",
        processing_method=None,
        moving_average=True,
    )
    list_channels = []
    # Create all channels possible
    channel_1 = Channel(
        "Single", no_channel=1, amplitude=20, pulse_width=250, frequency=25, name="Gastro",
        device_type=Device.Rehastimp24
    )
    stimulator = St(port="COM5")
    list_channels.append(channel_1)
    stimulator.init_stimulation(list_channels=list_channels)
    time_to_sleep = 0.001
    sign = 1
    while True:
        tic = time()
        packet = await interface.Connect.get_current_frame(components=interface.component)
        data = interface.get_force_plate_data(packet=packet)
        if len(data) != 0:
            force_ap_tmp = [data[0, 1, -10:]]
            plot_curve.update(np.append(force_ap_tmp, force_ap_tmp, axis=0))
            if np.mean(force_ap_tmp) < -10:
                new_sign = 0
            else:
                new_sign = 1

            if new_sign == 0 and sign != 0:
                stimulator.start_stimulation(upd_list_channels=list_channels, safety=True, stimulation_duration=0.2)
                print('send stim')
            elif new_sign == 1 and sign != 1:
                stimulator.pause_stimulation()

            sign = new_sign
        loop_time = time() - tic
        real_time_to_sleep = time_to_sleep - loop_time
        if real_time_to_sleep > 0:
            sleep(time_to_sleep - loop_time)





if __name__ == "__main__":
    asyncio.run(stim())

    print("All processes complete.")
    stimulator.end_stimulation()
    stimulator.close_port()
