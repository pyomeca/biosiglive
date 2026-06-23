import time
from pyScienceMode import Channel, Point, Device, Modes
from pyScienceMode import RehastimP24 as St

"""
This example shows how to use the RehastimP24 device. 
There are several commands divided into three levels (general, low and mid).
You can't call commands from different levels, you must first close the current level
to be able to use commands from another one.
"""
list_channels = []
# Create all channels possible

channel_1 = Channel("Single", no_channel=1, amplitude=8, pulse_width=250, frequency=25, name="Gastro",device_type=Device.Rehastimp24)
stimulator = St(port="COM5")
list_channels.append(channel_1)
stimulator.init_stimulation(list_channels=list_channels)

stimulator.start_stimulation(upd_list_channels = list_channels, safety = True, stimulation_duration = 0.1)


stimulator.close_port()
