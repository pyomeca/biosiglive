
from typing import Union


try:
    from pythonosc.udp_client import SimpleUDPClient

    osc_package = True
except ModuleNotFoundError:
    osc_package = False

try:
    import pytrigno
except ModuleNotFoundError:
    pass


from time import time, sleep, strftime
import datetime
import scipy.io as sio
import numpy as np
import multiprocessing as mp
import os
from biosiglive.streaming.server import Server
from biosiglive.io import save_data
from biosiglive.interfaces import pytrigno_interface, vicon_interface
from biosiglive.processing.data_processing import RealTimeProcessing
from biosiglive.processing.msk_functions import compute_inverse_kinematics
from biosiglive.gui.plot import LivePlot
from biosiglive.io.save_data import read_data
from ..interfaces.param import Device, MarkerSet
from ..enums import DeviceType, InterfaceType, InverseKinematicsMethods
from ..gui.plot import LivePlot
vicon_package, biorbd_package = True, True

try:
    import biorbd
except ModuleNotFoundError:
    biorbd_package = False

try:
    from vicon_dssdk import ViconDataStream as VDS
except ModuleNotFoundError:
    vicon_package = False


class StreamData:
    def __init__(self):
        self.process = mp.Process
        self.pool = mp.Pool
        self.queue = mp.Queue
        self.event = mp.Event
        self.devices = []
        self.devices_processing = []
        self.markers = []
        self.plots = []
        self.stream_rate = 100
        self.interfaces_type = []
        self.processes = []
        self.devices_processing_key = []
        self.markers_processing_key = []
        self.multiprocess = None

        # Multiprocessing stuff
        self.queue = mp.Queue()
        self.event = mp.Event()
        self.device_queue_in = []
        self.device_queue_out = []
        self.kin_queue_in = []
        self.kin_queue_out = []
        self.plots_queue_in = []
        self.plots_queue_out = []
        self.server_queue_in = []
        self.server_queue_out = []
        self.device_event = []
        self.interface_event = []
        self.marker_event = []
        self.custom_processes = []
        self.custom_processes_kwargs = []
        self.custom_processes_names = []
        self.custom_queue_in = []
        self.custom_queue_out = []
        self.custom_event = []
        self.models = []
        self.ik_methods = []
        self.save_data = None
        self.save_path = None
        self.save_frequency = None

        # Server stuff
        self.start_server = None
        self.server_ip = None
        self.ports = []
        self.client_type = None
        self.count_server = 0
        self.server_queue = []
        if isinstance(self.ports, int):
            self.ports = [self.ports]
        for p in range(len(self.ports)):
            self.server_queue.append(self.queue)

    def add_device(self, device: Device):
        self.devices.append(device)
        self.devices_processing_key.append(None)
        if device.interface not in self.interfaces_type:
            self.interfaces_type.append(device.interface)
            if self.multiprocess:
                self.interface_event.append(self.event)
        if self.multiprocess:
            self.device_queue_in.append(None)
            self.device_queue_out.append(None)
            self.device_event.append(None)

    def add_server(self, server_ip: str = "127.0.0.1", ports: Union[int, list] = 50000, client_type: str = "TCP"):
        self.server_ip = server_ip
        self.ports = ports
        self.client_type = client_type

    def start(self, save_streamed_data: bool = False, save_path: str = None, save_frequency: int = 100):
        self.save_data = save_streamed_data
        self.save_path = save_path
        self.save_frequency = save_frequency
        self._init_multiprocessing()

    def set_device_process_method(self, device_name: str, process_method: callable, **kwargs):
        device_idx = [device.name for device in self.devices].index(device_name)
        self.devices[device_idx].process_method = process_method
        self.devices_processing_key[device_idx] = kwargs
        if self.multiprocess:
            self.device_queue_in[device_idx] = self.queue
            self.device_queue_out[device_idx] = self.queue
            self.device_event.append(self.event)

    def set_kinematics_reconstruction_from_markers(self, model: str, marker_set_name: str, process_method: callable, **kwargs):
        marker_idx = [marker.name for marker in self.markers].index(marker_set_name)
        self.models[marker_idx] = model
        self.markers_processing_key[marker_idx] = kwargs
        self.ik_methods[marker_idx] = process_method
        if self.multiprocess:
            self.kin_queue_in[marker_idx] = self.queue
            self.kin_queue_out[marker_idx] = self.queue

    def add_marker_set(self, marker: MarkerSet):
        self.markers.append(marker)
        self.models.append(None)
        self.ik_methods.append(None)
        if marker.interface not in self.interfaces_type:
            self.interfaces_type.append(marker.interface)
            if self.multiprocess:
                self.interface_event.append(self.event)
        if self.multiprocess:
            self.kin_queue_in.append(None)
            self.kin_queue_out.append(None)

    def upd_marker_set(self, marker: MarkerSet, idx: int):
        self.markers[idx] = marker
        if marker.interface not in self.interfaces_type:
            self.interfaces_type.append(marker.interface)

    def upd_device(self, device: Device, idx: int):
        self.devices[idx] = device
        if device.interface not in self.interfaces_type:
            self.interfaces_type.append(device.interface)

    def device_processing(self, device: Device, device_idx : int, **kwargs):
        """
        Process the data from the device
        Parameters
        ----------
        device: Device
            The device to process
        device_idx: int
            The index of the device in the list of devices
        kwargs: dict
            The kwargs to pass to the process method
        Returns
        -------

        """
        if device.process_method is None:
            raise ValueError("No processing method defined for this device.")
        process_method = device.get_process_method()
        device_data = {}
        while True:
            try:
                device_data = self.device_queue_in[device_idx].get_nowait()
                is_working = True
            except mp.Queue().empty:
                is_working = False

            if is_working:
                device_data = device_data["device_data"]
                processed_data = process_method(device_data, **kwargs)
                self.device_queue_out[device_idx].put({"processed_data": processed_data})
                self.device_event[device_idx].set()

    def recons_kin(self, marker_idx: int, kalman_frequency: int = 100, processing_windows: int = 100, smoothing: bool = True):
        """
        Reconstruct kinematics from markers.
        Parameters
        ----------
        marker_idx: int
            Index of the marker set in the list of markers.
        kalman_frequency: int
            Frequency of the Kalman filter.
        processing_windows: int
            Size of the processing windows.
        smoothing: bool
            If True, the data will be smoothed in case of occlusion.
        Returns
        -------

        """
        model = biorbd.Model(self.models[marker_idx])
        markers_data = {}
        freq = kalman_frequency  # Hz
        params = biorbd.KalmanParam(freq)
        kalman = biorbd.KalmanReconsMarkers(model, params)
        while True:
            # If the queue is empty, the process is not working. Cannot use is_empty function as it is not trustable.
            try:
                markers_data = self.kin_queue_in[marker_idx].get_nowait()
                is_working = True
            except mp.Queue().empty:
                is_working = False
            if is_working:
                markers = markers_data["markers"]
                states = markers_data["states"]
                markers_tmp = markers_data["markers_tmp"]
                if self.ik_methods[marker_idx]:
                    q_tmp, dq_tmp = compute_inverse_kinematics(markers, model, self.ik_methods[marker_idx], kalman_frequency, kalman)
                    states_tmp = np.concatenate((q_tmp, dq_tmp), axis=0)
                    if len(states) == 0:
                        states = states_tmp
                    else:
                        if states.shape[1] < processing_windows:
                            states = np.append(states, states_tmp, axis=1)
                        else:
                            states = np.append(states[:, 1:], states_tmp, axis=1)

                if len(markers) != 0:
                    if smoothing:
                        for i in range(markers_tmp.shape[1]):
                            if np.product(markers_tmp[:, i, :]) == 0:
                                markers_tmp[:, i, :] = markers[:, i, -1:]
                if len(markers) == 0:
                    markers = markers_tmp
                else:
                    if markers.shape[2] < processing_windows:
                        markers = np.append(markers, markers_tmp, axis=2)
                    else:
                        markers = np.append(markers[:, :, 1:], markers_tmp[:, :, -1:], axis=2)
                self.kin_queue_out[marker_idx].put({"states": states, "markers": markers})
                self.marker_event[marker_idx].set()

    def open_server(self):
        """
        Open the server to send data from the devices.
        """
        server = Server(self.server_ip, self.ports[self.count_server], server_type=self.client_type)
        server.start()
        while True:
            connection, message = server.client_listening()
            data_queue = []
            while len(data_queue) == 0:
                try:
                    data_queue = self.server_queue[self.count_server].get_nowait()
                    is_working = True
                except mp.Queue().empty:
                    is_working = False
                if is_working:  # use this method to avoid blocking the server with Windows os
                    server.send_data(data_queue, connection, message)

    def _init_multiprocessing(self):
        """
        Initialize the multiprocessing.
        """
        processes = []
        for d, device in enumerate(self.devices):
            if device.process_method is not None:
                if not self.devices_processing_key[d]:
                    raise ValueError("No processing method defined for this device. "
                                     "Use set_device_process_method to define it.")
                self.processes.append(self.process(name=f"process_{device.name}",
                                                   target=StreamData.device_processing, args=(self, device, d,)))
        if self.start_server:
            for i in range(len(self.ports)):
                processes.append(self.process(name="listen" + f"_{i}", target=StreamData.open_server, args=(self,)))
                self.count_server += 1

        processes = [self.process(name="reader", target=LiveData.save_streamed_data, args=(self,))]
        for m, marker in enumerate(self.markers):
            processes.append(self.process(name=f"process_{marker.name}", target=StreamData.recons_kin, args=(self, marker, m,)))

        for i, funct in enumerate(self.custom_processes):
            processes.append(self.process(name=self.custom_processes_names[i], target=funct, args=(self,), kwargs=self.custom_processes_kwargs[i]))
        for p in processes:
            p.start()
        self.multiprocess_started = True
        for p in processes:
            p.join()

    def add_custom_process(self, funct: callable, name: str, **kwargs):
        """
        Add a custom process to the multiprocessing.
        Parameters
        ----------
        funct: callable
            Function to add to the multiprocessing.
        name: str
            Name of the process.
        kwargs: dict
            Keyword arguments of the function.
        """
        if name in [p.name for p in self.custom_processes]:
            name = name + str(len(self.custom_processes))
        self.custom_processes_names.append(name)
        if self.multiprocess_started:
            raise ValueError("Cannot add custom process after starting the stream.")
        self.custom_processes.append(funct)
        self.custom_processes_kwargs.append(kwargs)

    def add_plot(self, plot: LivePlot, data_name:str, plot_rate: int = None, plot_windows: int = 100):
        """
        Add a plot to the live data.
        Parameters
        ----------
        plot: LivePlot
            Plot to add.
        plot_rate: int
            Rate of the plot.

        """
        plot_rate = self.stream_rate if not plot_rate else plot_rate
        if plot_rate > self.stream_rate:
            raise ValueError("Plot rate cannot be higher than stream rate.")
        self.plots.append(plot)

    def plot_update(self):
        """
        Update the plots.
        """
        for plot in self.plots:
            plot.update()

    def save_streamed_data(self):
        raw_emg = []
        raw_imu = []
        imu_proc = []
        emg_proc = []
        markers = []
        states = []
        vicon_latency_total = 0
        initial_time = 0
        absolute_time_frame = 0
        if self.try_w_connection:
            self.interface.init_client()
        # self.nb_marks = len(self.marker_names)
        delta = 0
        delta_tmp = 0
        self.iter = 0
        dic_to_put = {}
        c = 0
        m = 0
        while True:
            tic = time()
            if self.iter == 0:
                initial_time = time() - tic
            if self.try_w_connection:
                self.interface.get_frame()
                is_frame = self.interface.is_frame()
                vicon_latency_total = self.interface.get_latency()
            else:
                is_frame = True
            absolute_time_frame = datetime.datetime.now()
            absolute_time_frame_dic = {
                "day": absolute_time_frame.day,
                "hour": absolute_time_frame.hour,
                "hour_s": absolute_time_frame.hour * 3600,
                "minute": absolute_time_frame.minute,
                "minute_s": absolute_time_frame.minute * 60,
                "second": absolute_time_frame.second,
                "millisecond": int(absolute_time_frame.microsecond / 1000),
                "millisecond_s": int(absolute_time_frame.microsecond / 1000) * 0.001,
            }

            if is_frame:
                if self.iter == 0:
                    print("Data start streaming")
                    self.iter = 1
                if self.stream_emg:
                    if self.try_w_connection:
                        all_device_data = self.interface.get_device_data("all")
                        for i, device in enumerate(self.interface.devices):
                            if device.type == "emg":
                                emg_tmp = all_device_data[i][:10, :]
                    else:
                        emg_tmp = self.emg_exp[: self.nb_electrodes, c : c + self.emg_sample]
                        c = c + self.emg_sample
                        # c = c + self.emg_sample if c + self.emg_sample < self.last_frame else self.init_frame
                    self.emg_queue_in.put_nowait({"raw_emg": raw_emg, "emg_proc": emg_proc, "emg_tmp": emg_tmp})
                if self.stream_markers:
                    if self.try_w_connection:
                        all_markers_tmp, all_occluded = self.interface.get_markers_data()
                        markers_tmp = all_markers_tmp[0]
                    else:
                        markers_tmp = self.markers_exp[:, :, m : m + 1]
                        m = m + 1
                        # m = m + 1 if m < self.last_frame else self.init_frame
                    self.kin_queue_in.put_nowait(
                        {
                            "states": states,
                            "markers": markers,
                            "model_path": self.model_path,
                            "markers_tmp": markers_tmp,
                        }
                    )

                if self.stream_emg:
                    self.event_emg.wait()
                    emg_data = self.emg_queue_out.get_nowait()
                    self.event_emg.clear()
                    raw_emg, emg_proc = emg_data["raw_emg"], emg_data["emg_proc"]
                    # dic_to_put["emg_names"] = emg_names
                    dic_to_put["raw_emg"] = np.around(raw_emg, decimals=self.emg_dec)
                    dic_to_put["emg_proc"] = np.around(emg_proc, decimals=self.emg_dec)
                    dic_to_put["emg_sample"] = self.emg_sample

                if self.stream_markers:
                    self.event_kin.wait()
                    kin = self.kin_queue_out.get_nowait()
                    self.event_kin.clear()
                    states, markers = kin["states"], kin["markers"]
                    dic_to_put["markers"] = np.around(markers, decimals=self.markers_dec)
                    dic_to_put["kalman"] = states
                    dic_to_put["marker_names"] = self.marker_names

                dic_to_put["acquisition_rate"] = self.acquisition_rate
                dic_to_put["absolute_time_frame"] = absolute_time_frame_dic
                dic_to_put["vicon_latency"] = vicon_latency_total
                process_time = time() - tic  # time to process all data + time to get data
                for i in range(len(self.server_ports)):
                    try:
                        self.server_queue[i].get_nowait()
                    except:
                        pass
                    self.server_queue[i].put_nowait(dic_to_put)
                self.iter += 1

                # Save data
                if self.save_data is True:
                    data_to_save = {
                        "process_delay": process_time,
                        "absolute_time_frame": absolute_time_frame_dic,
                        "vicon_latency_total": vicon_latency_total,
                        "initial_time": initial_time,
                        "emg_freq": self.emg_rate,
                        "acquisition_freq": self.acquisition_rate,
                    }
                    if self.stream_emg:
                        data_to_save["emg_proc"] = emg_proc[:, -1:]
                        data_to_save["raw_emg"] = raw_emg[:, -self.emg_sample :]

                    if self.stream_markers:
                        data_to_save["markers"] = markers[:3, :, -1:]

                    if self.recons_kalman:
                        data_to_save["kalman"] = states[:, -1:]
                    save_data.add_data_to_pickle(data_to_save, self.output_file_path)

                self.iter += 1
            print(time() - tic)
            if not self.try_w_connection:
                sleep(1 / self.acquisition_rate)





