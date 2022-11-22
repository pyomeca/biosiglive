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
from biosiglive.processing.msk_functions import MskFunctions
from ..interfaces.generic_interface import GenericInterface
from biosiglive.gui.plot import LivePlot
from biosiglive.io.save_data import read_data
from ..interfaces.param import Device, MarkerSet
from ..enums import InterfaceType, DeviceType, MarkerType, InverseKinematicsMethods, RealTimeProcessingMethod
from ..gui.plot import LivePlot
from .utils import dic_merger

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
    def __init__(self, stream_rate: int = 100, multiprocess: bool = True):
        """
        Initialize the StreamData class.
        Parameters
        ----------
        stream_rate: int
            The stream rate of the data.
        multiprocess:
            If True, the data will be processed, disseminate and plot in separate processes.
        """
        self.process = mp.Process
        self.pool = mp.Pool
        self.queue = mp.Queue
        self.event = mp.Event
        self.devices = []
        self.devices_processing = []
        self.marker_sets = []
        self.plots = []
        self.stream_rate = stream_rate
        self.interfaces_type = []
        self.processes = []
        self.devices_processing_key = []
        self.marker_sets_processing_key = []
        self.interfaces = []
        self.multiprocess = multiprocess

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

    def _add_device(self, device: Device):
        """
        Add a device to the stream.
        Parameters
        ----------
        device: Device
            Device to add.
        """
        self.devices.append(device)
        self.devices_processing_key.append(None)
        if self.multiprocess:
            self.device_queue_in.append(None)
            self.device_queue_out.append(None)
            self.device_event.append(None)

    def add_interface(self, interface: callable):
        """
        Add an interface to the stream.
        Parameters
        ----------
        interface: GenericInterface
            Interface to add. Interface should inherit from the generic interface.
        """
        if self.multiprocess_started:
            raise Exception("Cannot add interface after the stream has started.")
        self.interfaces.append(interface)
        self.interfaces_type.append(interface.interface_type)
        self.interface_event.append(self.event)
        for device in interface.devices:
            self._add_device(device)
        for marker in interface.markers:
            self._add_marker_set(marker)
        if len(self.interfaces) > 1:
            raise ValueError("Only one interface can be added for now.")

    def add_server(self, server_ip: str = "127.0.0.1", ports: Union[int, list] = 50000, client_type: str = "TCP"):
        """
        Add a server to the stream.
        Parameters
        ----------
        server_ip: str
            The ip address of the server.
        ports: int or list
            The port(s) of the server.
        client_type: str
            The type of client to use. Can be TCP.
        """
        if self.multiprocess_started:
            raise Exception("Cannot add interface after the stream has started.")
        self.server_ip = server_ip
        self.ports = ports
        self.client_type = client_type

    def start(self, save_streamed_data: bool = False, save_path: str = None, save_frequency: int = None):
        """
        Start the stream.
        Parameters
        ----------
        save_streamed_data: bool
            If True, the streamed data will be saved.
        save_path: str
            The path to save the streamed data.
        save_frequency:
            The frequency at which the data will be saved.
        """
        self.save_data = save_streamed_data
        self.save_path = save_path if save_path else f"streamed_data_{strftime('%Y%m%d_%H%M%S')}.bio"
        self.save_frequency = save_frequency if save_frequency else self.stream_rate
        self._init_multiprocessing()

    # def set_kinematics_reconstruction_from_markers(
    #     self, model: str, marker_set_name: str, process_method: callable, **kwargs
    # ):
    #     marker_idx = [marker.name for marker in self.marker_sets].index(marker_set_name)
    #     self.models[marker_idx] = model
    #     self.marker_sets_processing_key[marker_idx] = kwargs
    #     self.ik_methods[marker_idx] = process_method
    #     if self.multiprocess:
    #         self.kin_queue_in[marker_idx] = self.queue
    #         self.kin_queue_out[marker_idx] = self.queue

    def _add_marker_set(self, marker: MarkerSet):
        """
        Add a marker set to the stream.
        Parameters
        ----------
        marker: MarkerSet
            Marker set to add from given interface.
        """
        self.marker_sets.append(marker)
        self.models.append(None)
        self.ik_methods.append(None)
        if self.multiprocess:
            self.kin_queue_in.append(None)
            self.kin_queue_out.append(None)

    def device_processing(self, device: Device, device_idx: int, **kwargs):
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

    def recons_kin(
        self, marker_idx: int, kalman_frequency: int = 100, processing_windows: int = 100, smoothing: bool = True
    ):
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
                    q_tmp, dq_tmp = MskFunctions.compute_inverse_kinematics(
                        markers, model, self.ik_methods[marker_idx], kalman_frequency, kalman
                    )
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
                    raise ValueError(
                        "No processing method defined for this device. " "Use set_device_process_method to define it."
                    )
                self.processes.append(
                    self.process(
                        name=f"process_{device.name}",
                        target=StreamData.device_processing,
                        args=(
                            self,
                            device,
                            d,
                        ),
                    )
                )
        if self.start_server:
            for i in range(len(self.ports)):
                processes.append(self.process(name="listen" + f"_{i}", target=StreamData.open_server, args=(self,)))
                self.count_server += 1
        for interface in self.interfaces:
            processes.append(self.process(name="reader", target=StreamData.save_streamed_data, args=(self, interface)))
        for m, marker in enumerate(self.marker_sets):
            processes.append(
                self.process(
                    name=f"process_{marker.name}",
                    target=StreamData.recons_kin,
                    args=(
                        self,
                        marker,
                        m,
                    ),
                )
            )

        for i, funct in enumerate(self.custom_processes):
            processes.append(
                self.process(
                    name=self.custom_processes_names[i],
                    target=funct,
                    args=(self,),
                    kwargs=self.custom_processes_kwargs[i],
                )
            )
        for p in processes:
            p.start()
        self.multiprocess_started = True
        for p in processes:
            p.join()

    def add_plot(self, plot: Union[LivePlot, list]):
        """
        Add a plot to the live data.
        Parameters
        ----------
        plot: Union[LivePlot, list]
            Plot to add.
        """
        if self.multiprocess_started:
            raise Exception("Cannot add interface after the stream has started.")
        for plt in plot:
            if plt.rate:
                if plt.rate > self.stream_rate:
                    raise ValueError("Plot rate cannot be higher than stream rate.")
            self.plots.append(plt)

    def plot_update(self, data: Union[list, np.ndarray]):
        """
        Update the plots.

        Parameters
        ----------
        data: Union[list, np.ndarray]
            Data to plot. If list, length should be the number of total plots.
        """
        if isinstance(data, list):
            if len(data) != len(self.plots):
                raise ValueError("Data length should be the same as the number of plots.")
        if len(self.plots) == 1:
            if isinstance(data, np.ndarray):
                data = [data]
        for p, plot in enumerate(self.plots):
            plot.update(data[p])

    def save_streamed_data(self, interface: GenericInterface):
        """
        Stream, process and save the data.
        Parameters
        ----------
        interface: callable
            Interface to use to get the data.

        """
        initial_time = 0
        iteration = 0
        dic_to_save = {}
        save_count = 0
        self.save_frequency = self.save_frequency if self.save_frequency else self.stream_rate
        while True:
            data_dic = {}
            proc_device_data = []
            raw_device_data = []
            raw_markers_data = []
            all_device_data = []
            all_markers_tmp = []
            kin_data = []
            tic = time()
            if iteration == 0:
                initial_time = time() - tic
            interface_latency = interface.get_latency()
            is_frame = interface.get_frame()
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
                if iteration == 0:
                    print("Data start streaming")
                    iteration = 1
                if len(interface.devices) != 0:
                    all_device_data = interface.get_device_data(device_name="all", get_frame=False)
                    for i in range(len(interface.devices)):
                        self.device_queue_in[i].put_nowait({"device_data_tmp": all_device_data[i]})
                if len(interface.markers) != 0:
                    all_markers_tmp, _ = interface.get_marker_set_data(get_frame=False)
                    for i in range(len(interface.devices)):
                        self.kin_queue_in[i].put_nowait({"markers_tmp": all_markers_tmp[i]})
                time_to_get_data = time() - tic
                tic_process = time()
                if len(interface.devices) != 0:
                    for i in range(len(interface.devices)):
                        if self.devices[i].process_method is not None:
                            self.device_event[i].wait()
                            device_data = self.device_queue_out[i].get_nowait()
                            self.device_event[i].clear()
                            raw_device_data.append(np.around(device_data["raw_device_data"], decimals=self.device_decimal))
                            proc_device_data.append(np.around(device_data["proc_device_data"], decimals=self.device_decimal))
                    if len(raw_device_data) == 0:
                        raw_device_data = all_device_data
                    else:
                        data_dic["proc_device_data"] = proc_device_data
                data_dic["raw_device_data"] = raw_device_data

                if len(interface.markers) != 0:
                    for i in range(len(interface.markers)):
                        if self.marker_sets[i].process is not None:
                            self.marker_event[i].wait()
                            markers_data = self.kin_queue_out[i].get_nowait()
                            self.marker_event[i].clear()
                            raw_markers_data.append(np.around(markers_data["raw_markers_data"], decimals=self.kin_dec))
                            kin_data.append(np.around(markers_data["kinematics_data"], decimals=self.kin_dec))
                    if len(raw_markers_data) == 0:
                        raw_markers_data = all_markers_tmp
                    else:
                        data_dic["kinematics_data"] = kin_data
                    data_dic["raw_device_data"] = raw_markers_data
                process_time = time() - tic_process  # time to process all data + time to get data

                for i in range(len(self.ports)):
                    try:
                        self.server_queue[i].get_nowait()
                    except:
                        pass
                    self.server_queue[i].put_nowait(data_dic)

                data_dic["absolute_time_frame"] = absolute_time_frame_dic
                data_dic["interface_latency"] = interface_latency
                data_dic["process_time"] = process_time
                data_dic["initial_time"] = initial_time
                data_dic["time_to_get_data"] = time_to_get_data

                # Save data
                if self.save_data is True:
                    dic_to_save = dic_merger(data_dic, dic_to_save)
                    if save_count == int(self.stream_rate / self.save_frequency):
                        save_data.add_data_to_pickle(data_dic, self.save_path)
                        dic_to_save = {}
                        save_count = 0
                    save_count += 1
