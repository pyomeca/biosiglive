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
     vicon_interface = ViconClient(init_now=True)
     vicon_interface.add_device("Treadmill", "generic_device", rate=2000, system_rate=100)
     vicon_interface.devices[-1].set_process_method(RealTimeProcessing().get_peaks)
     nb_min_frame = vicon_interface.devices[-1].rate * 10
     time_to_sleep = 1 / vicon_interface.devices[-1].system_rate
     count = 0
     force_z, force_z_process = [], []
     is_one = [False, False]

     while True:
         tic = time()
         vicon_interface.get_frame()
         data = vicon_interface.get_device_data(device_name="Treadmill")
         force_z_tmp = data[0][[2, 8], :]
         cadence, force_z_process, force_z, is_one = vicon_interface.devices[0].process_method(new_sample=force_z_tmp,
                                                     signal=force_z,
                                                     signal_proc=force_z_process,
                                                     threshold=0.2,
                                                     nb_min_frame=nb_min_frame,
                                                     is_one=is_one,
                                                     min_peaks_interval=1300
                                                     )

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
     channel_1 = Ch.Channel('Single', no_channel=1, amplitude=50, pulse_width=100, stimulation_interval=33, name='Biceps')
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

     stimulator = St.Stimulator(list_channels, 'COM34')
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


if __name__ == '__main__':
     stimulation_delay = 10 # ms
     stimulation_duration = 0.33 # s
     foot_strike = mp.Event()
     stim_proc = mp.Process(name="stim", target=stim, args=(foot_strike, stimulation_delay, stimulation_duration,))
     stream_proc = mp.Process(name="stream", target=stream, args=(foot_strike,))
     stim_proc.start()
     stream_proc.start()
     stim_proc.join()
     stream_proc.join()
