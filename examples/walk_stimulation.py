import numpy as np
from biosiglive.interfaces.vicon_interface import ViconClient
from biosiglive.processing.data_processing import RealTimeProcessing
from biosiglive.gui.plot import LivePlot
from time import sleep, time

# Import Stimulator class
from pyScienceMode2 import Stimulator as St

# Import Channel class
from pyScienceMode2 import Channel as Ch
import multiprocessing as mp


def stream(foot_strike):
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
        interface.get_device_data(device_name="Treadmill")
        cadence, force_z_process = interface.get_device("Treadmill").process()
        if np.count_nonzero(force_z_process[:, -20:]):
            print("set")
            foot_strike.set()
            count += 1
        loop_time = time() - tic
        real_time_to_sleep = time_to_sleep - loop_time
        if real_time_to_sleep > 0:
            sleep(time_to_sleep - loop_time)


def stim(foot_strike, stimulation_delay, stimulation_duration):
    list_channels = []

    # Create all channels possible
    channel_1 = Ch.Channel(
        "Single", no_channel=1, amplitude=50, pulse_width=100, stimulation_interval=33, name="Biceps"
    )
    # channel_2 = Ch.Channel()
    # channel_2.set_mode('Single')
    # channel_2.set_no_channel(3)
    # channel_2.set_amplitude(2)
    # channel_2.set_frequency(20)
    # channel_2.set_pulse_width(100)
    # channel_2.set_stimulation_interval(100)
    # channel_2.set_inter_pulse_interval(10)
    # channel_2.set_name('Triceps')
    list_channels.append(channel_1)
    # list_channels.append(channel_2)

    stimulator = St.Stimulator(list_channels, "COM34")
    # Show the log, by default True if called, to deactivate : stimulator.show_log(False)
    # stimulator.show_log()
    count = 0
    while True:
        foot_strike.wait()
        sleep(stimulation_delay * 0.001)
        # if count == 0:
        # Si on utilise la function start_stimulation avec un temps défini, alors elle appelle stop_stimulation()
        # après ce temps. Il faut donc relancer start_stimulation.
        stimulator.start_stimulation(stimulation_duration)
        # sleep(stimulation_duration)
        # count += 1
        # else:
        # stimulator.update_stimulation(list_channels)
        # sleep(stimulation_duration)
        # print(round(time(), 3))
        # stimulator.stop_stimulation()
        print("stim_started")
        foot_strike.clear()


if __name__ == "__main__":
    stimulation_delay = 10  # ms
    stimulation_duration = 0.33  # s
    foot_strike = mp.Event()
    stim_proc = mp.Process(
        name="stim",
        target=stim,
        args=(
            foot_strike,
            stimulation_delay,
            stimulation_duration,
        ),
    )
    stream_proc = mp.Process(name="stream", target=stream, args=(foot_strike,))
    stim_proc.start()
    stream_proc.start()
    stim_proc.join()
    stream_proc.join()
