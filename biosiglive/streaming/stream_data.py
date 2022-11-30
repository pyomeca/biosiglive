from typing import Union
from time import time, sleep, strftime
import datetime
import numpy as np
import multiprocessing as mp
from biosiglive.streaming.server import Server
from ..file_io.save_and_load import save
from ..interfaces.generic_interface import GenericInterface
from ..interfaces.param import Device, MarkerSet
from ..gui.plot import LivePlot
from .utils import dic_merger


# TODO add enum for command type
class StreamData:
    def __init__(self, stream_rate: int = 100):
        """
        Initialize the StreamData class.
        Parameters
        ----------
        stream_rate: int
            The stream rate of the data.
        """
        self.process = mp.Process
        self.devices = []
        self.marker_sets = []
        self.plots = []
        self.stream_rate = stream_rate
        self.interfaces_type = []
        self.processes = []
        self.interfaces = []
        self.multiprocess_started = False

        # Multiprocessing stuff
        manager = mp.Manager()
        self.queue = manager.Queue
        self.event = manager.Event
        self.device_queue_in = []
        self.device_queue_out = []
        self.kin_queue_in = []
        self.kin_queue_out = []
        self.plots_queue = []
        self.device_event = []
        self.is_device_data = self.event()
        self.is_kin_data = self.event()
        self.interface_event = []
        self.kin_event = []
        self.custom_processes = []
        self.custom_processes_kwargs = []
        self.custom_processes_names = []
        self.custom_queue_in = []
        self.custom_queue_out = []
        self.custom_event = []
        self.save_data = None
        self.save_path = None
        self.save_frequency = None
        self.plots_multiprocess = False
        self.device_buffer_size = []
        self.marker_set_buffer_size = []
        self.raw_plot = None
        self.data_to_plot = None

        # Server stuff
        self.start_server = None
        self.server_ip = None
        self.ports = []
        self.client_type = None
        self.count_server = 0
        self.server_queue = []
        self.device_decimals = 8
        self.kin_decimals = 6

    def _add_device(self, device: Device):
        """
        Add a device to the stream.
        Parameters
        ----------
        device: Device
            Device to add.
        """
        self.devices.append(device)
        self.device_queue_in.append(self.queue())
        self.device_queue_out.append(self.queue())
        self.device_event.append(self.event())

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
        self.interface_event.append(self.event())
        for device in interface.devices:
            self._add_device(device)
        for marker in interface.marker_sets:
            self._add_marker_set(marker)
        if len(self.interfaces) > 1:
            raise ValueError("Only one interface can be added for now.")

    def add_server(
        self,
        server_ip: str = "127.0.0.1",
        ports: Union[int, list] = 50000,
        client_type: str = "TCP",
        device_buffer_size: Union[int, list] = None,
        marker_set_buffer_size: [int, list] = None,
    ):
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
        device_buffer_size: int or list
            The size of the buffer for the devices.
        marker_set_buffer_size: int or list
            The size of the buffer for the marker sets.
        """
        if self.multiprocess_started:
            raise Exception("Cannot add interface after the stream has started.")
        self.server_ip = server_ip
        self.ports = ports
        if not isinstance(self.ports, list):
            self.ports = [self.ports]

        for p in range(len(self.ports)):
            self.server_queue.append(self.queue())
        self.client_type = client_type

        if not device_buffer_size:
            device_buffer_size = [None] * len(self.devices)
        if isinstance(device_buffer_size, list):
            if len(device_buffer_size) != len(self.devices):
                raise ValueError("The device buffer size list should have the same length as the number of devices.")
            self.device_buffer_size = device_buffer_size
        elif isinstance(device_buffer_size, int):
            self.device_buffer_size = [device_buffer_size] * len(self.devices)

        if not marker_set_buffer_size:
            marker_set_buffer_size = [None] * len(self.marker_sets)
        if isinstance(marker_set_buffer_size, list):
            if len(marker_set_buffer_size) != len(self.marker_sets):
                raise ValueError(
                    "The marker set buffer size list should have the same length as the number of marker sets."
                )
            self.marker_set_buffer_size = marker_set_buffer_size
        elif isinstance(marker_set_buffer_size, int):
            self.marker_set_buffer_size = [marker_set_buffer_size] * len(self.marker_sets)

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

    def _add_marker_set(self, marker: MarkerSet):
        """
        Add a marker set to the stream.
        Parameters
        ----------
        marker: MarkerSet
            Marker set to add from given interface.
        """
        self.marker_sets.append(marker)
        self.kin_queue_in.append(self.queue())
        self.kin_queue_out.append(self.queue())
        self.kin_event.append(self.event())

    # TODO : add buffer directly in the server
    def device_processing(self, device_idx: int):
        """
        Process the data from the device
        Parameters
        ----------
        device_idx: int
            The index of the device in the list of devices
        kwargs: dict
            The kwargs to pass to the process method
        Returns
        -------

        """
        if self.device_buffer_size:
            if not self.device_buffer_size[device_idx]:
                self.device_buffer_size[device_idx] = self.devices[device_idx].rate
            buffer_size = self.device_buffer_size[device_idx]
        else:
            buffer_size = self.devices[device_idx].rate
        device_data = []
        while True:
            self.is_device_data.wait()
            try:
                device_data = self.device_queue_in[device_idx].get_nowait()
                is_working = True
            except Exception:
                is_working = False
            if is_working:
                self.devices[device_idx].new_data = device_data
                self.devices[device_idx].append_data(device_data)
                processed_data = self.devices[device_idx].process(**self.devices[device_idx].processing_method_kwargs)
                self.device_queue_out[device_idx].put_nowait({"processed_data": processed_data[:, -buffer_size:]})
                self.device_event[device_idx].set()

    def recons_kin(self, marker_set_idx: int):
        """
        Reconstruct kinematics from markers.
        Parameters
        ----------
        marker_set_idx: int
            Index of the marker set in the list of markers.
        Returns
        -------

        """
        if self.marker_set_buffer_size:
            if not self.marker_set_buffer_size[marker_set_idx]:
                self.marker_set_buffer_size[marker_set_idx] = self.marker_sets[marker_set_idx].rate
            buffer_size = self.marker_set_buffer_size[marker_set_idx]
        else:
            buffer_size = self.marker_sets[marker_set_idx].rate
        if "model_path" not in self.marker_sets[marker_set_idx].kin_method_kwargs.keys():
            raise ValueError("No model to compute the kinematics.")
        markers = []
        while True:
            self.is_kin_data.wait()
            try:
                markers = self.kin_queue_in[marker_set_idx].get_nowait()
                is_working = True
            except Exception:
                is_working = False
            if is_working:
                self.marker_sets[marker_set_idx].new_data = markers
                self.marker_sets[marker_set_idx].append_data(markers)
                states, _ = self.marker_sets[marker_set_idx].get_kinematics(
                    **self.marker_sets[marker_set_idx].kin_method_kwargs
                )
                self.kin_queue_out[marker_set_idx].put_nowait({"kinematics_data": states[:, -buffer_size:]})
                self.kin_event[marker_set_idx].set()

    def open_server(self, server_idx: int):
        """
        Open the server to send data from the devices.
        """
        server = Server(self.server_ip, self.ports[server_idx], server_type=self.client_type)
        server.start()
        while True:
            connection, message = server.client_listening()
            data_queue = []
            while len(data_queue) == 0:
                # use Try statement as the queue can be empty and is_empty function is not reliable.
                try:
                    data_queue = self.server_queue[server_idx].get_nowait()
                    is_working = True
                except Exception:
                    is_working = False
                if is_working:  # use this method to avoid blocking the server with Windows os.
                    server.send_data(data_queue, connection, message)

    def _init_multiprocessing(self):
        """
        Initialize the multiprocessing.
        """
        processes = []
        for i in range(len(self.interfaces)):
            processes.append(
                self.process(name="reader", target=StreamData.save_streamed_data, args=(self, i), daemon=True)
            )

        for d, device in enumerate(self.devices):
            if device.processing_method is not None:
                processes.append(
                    self.process(
                        name=f"process_{device.name}",
                        target=StreamData.device_processing,
                        args=(
                            self,
                            d,
                        ),
                        daemon=True,
                    )
                )
        for i in range(len(self.ports)):
            processes.append(
                self.process(name="listen" + f"_{i}", target=StreamData.open_server, args=(self, i), daemon=True)
            )

        for p, plot in enumerate(self.plots):
            for device in self.devices:
                for marker_set in self.marker_sets:
                    if self.data_to_plot[p] not in device.name and self.data_to_plot[p] not in marker_set.name:
                        raise ValueError(f"The name of the data to plot ({self.data_to_plot[p]}) is not correct.")
            if self.plots_multiprocess:
                processes.append(self.process(name="plot", target=StreamData.plot_update, args=(self, p), daemon=True))
            else:
                processes.append(self.process(name="plot", target=StreamData.plot_update, args=(self, -1), daemon=True))
                break

        for m, marker in enumerate(self.marker_sets):
            if marker.kin_method:
                processes.append(
                    self.process(
                        name=f"process_{marker.name}",
                        target=StreamData.recons_kin,
                        args=(
                            self,
                            m,
                        ),
                        daemon=True,
                    )
                )

        for i, funct in enumerate(self.custom_processes):
            processes.append(
                self.process(
                    name=self.custom_processes_names[i],
                    target=funct,
                    args=(self,),
                    kwargs=self.custom_processes_kwargs[i],
                    daemon=True,
                )
            )
        for p in processes:
            p.start()
        self.multiprocess_started = True
        for p in processes:
            p.join()

    def _check_nb_processes(self):
        """
        compute the number of process.
        """
        nb_processes = 0
        for device in self.devices:
            if device.process_method is not None:
                nb_processes += 1
        if self.start_server:
            nb_processes += len(self.ports)
        nb_processes += len(self.plots)
        nb_processes += len(self.interfaces)
        for marker in self.marker_sets:
            if marker.kin_method:
                nb_processes += 1
        nb_processes += len(self.custom_processes)
        return nb_processes

    def add_plot(
        self,
        plot: Union[LivePlot, list],
        data_to_plot: Union[str, list],
        raw: Union[bool, list] = None,
        multiprocess=False,
    ):
        """
        Add a plot to the live data.
        Parameters
        ----------
        plot: Union[LivePlot, list]
            Plot to add.
        data_to_plot: Union[str, list]
            Name of the data to plot.
        raw: Union[bool, list]
            If True, the raw data will be plotted.
        multiprocess: bool
            If True, if several plot each plot will be on a separate process. If False, each plot will be on the same one.
        """
        raise NotImplementedError("Plot are not implemented yet with StreamData class.")
        # if isinstance(data_to_plot, str):
        #     data_to_plot = [data_to_plot]
        # if isinstance(raw, bool):
        #     raw = [raw]
        # if len(data_to_plot) != len(raw):
        #     raise ValueError("The length of the data to plot and the raw list must be the same.")
        # if not raw:
        #     raw = [True] * len(data_to_plot)
        # self.plots_queue.append(self.queue())
        # self.raw_plot = raw
        # self.data_to_plot = data_to_plot
        # if self.multiprocess_started:
        #     raise Exception("Cannot add plot after the stream has started.")
        # self.plots_multiprocess = multiprocess
        # if not isinstance(plot, list):
        #     plot = [plot]
        # for plt in plot:
        #     if plt.rate:
        #         if plt.rate > self.stream_rate:
        #             raise ValueError("Plot rate cannot be higher than stream rate.")
        #     self.plots.append(plt)

    def plot_update(self, plot_idx: int = -1):
        """
        Update the plots.

        Parameters
        ----------
        plot_idx: int
            index of the plot to update. If -1, all plots will be updated.
        """
        if plot_idx == -1:
            plots = self.plots
            queue = self.plots_queue[0]
        else:
            plots = self.plots[plot_idx]
            queue = self.plots_queue[plot_idx]
        data_to_plot = []
        data = None
        device_names = []
        marker_set_names = []
        for device in self.devices:
            device_names.append(device.name)
        for marker in self.marker_sets:
            marker_set_names.append(marker.name)
        while True:
            try:
                data = queue.get_nowait()
                is_working = True
            except Exception:
                is_working = False
            if is_working:
                for p, plot in enumerate(plots):
                    if self.data_to_plot[p] in device_names:
                        if not self.raw_plot[p]:
                            data_to_plot = data["proc_device_data"][device_names.index(self.data_to_plot[p])]
                        else:
                            data_to_plot = data["raw_device_data"][device_names.index(self.data_to_plot[p])]
                    if self.data_to_plot[p] in marker_set_names:
                        if not self.raw_plot[p]:
                            data_to_plot = data["kinematics_data"][marker_set_names.index(self.data_to_plot[p])]
                        else:
                            data_to_plot = data["marker_set_data"][marker_set_names.index(self.data_to_plot[p])][
                                :, :, -1
                            ].T
                    plot.update(data_to_plot)

    def save_streamed_data(self, interface_idx: int):
        """
        Stream, process and save the data.
        Parameters
        ----------
        interface_idx: idx
            Interface idx to use.

        """
        initial_time = 0
        iteration = 0
        dic_to_save = {}
        save_count = 0
        self.save_frequency = self.save_frequency if self.save_frequency else self.stream_rate
        interface = self.interfaces[interface_idx]
        saving_time = None
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
            self.is_kin_data.clear()
            self.is_device_data.clear()
            if is_frame:
                if iteration == 0:
                    print("Data start streaming")
                    iteration = 1
                if len(interface.devices) != 0:
                    all_device_data = interface.get_device_data(device_name="all", get_frame=False)
                    if not isinstance(all_device_data, list):
                        all_device_data = [all_device_data]
                    self.is_device_data.set()
                    for i in range(len(all_device_data)):
                        if self.devices[i].processing_method is not None:
                            self.device_queue_in[i].put_nowait(all_device_data[i])
                if len(interface.marker_sets) != 0:
                    all_markers_tmp, _ = interface.get_marker_set_data(get_frame=False)
                    if not isinstance(all_markers_tmp, list):
                        all_markers_tmp = [all_markers_tmp]
                    self.is_kin_data.set()
                    for i in range(len(self.marker_sets)):
                        if self.marker_sets[i].kin_method is not None:
                            self.kin_queue_in[i].put_nowait(all_markers_tmp[i])
                time_to_get_data = time() - tic
                tic_process = time()
                if len(interface.devices) != 0:
                    for i in range(len(interface.devices)):
                        if self.devices[i].processing_method is not None:
                            self.device_event[i].wait()
                            device_data = self.device_queue_out[i].get_nowait()
                            self.device_event[i].clear()
                            proc_device_data.append(
                                np.around(device_data["processed_data"], decimals=self.device_decimals)
                            )
                        raw_device_data.append(np.around(all_device_data[i], decimals=self.device_decimals))
                    data_dic["proc_device_data"] = proc_device_data
                    data_dic["raw_device_data"] = raw_device_data

                if len(interface.marker_sets) != 0:
                    for i in range(len(interface.marker_sets)):
                        if self.marker_sets[i].kin_method is not None:
                            self.kin_event[i].wait()
                            kin_data_proc = self.kin_queue_out[i].get_nowait()
                            self.kin_event[i].clear()
                            kin_data.append(np.around(kin_data_proc["kinematics_data"], decimals=self.kin_decimals))
                        raw_markers_data.append(np.around(all_markers_tmp[i], decimals=self.kin_decimals))
                    data_dic["kinematics_data"] = kin_data
                    data_dic["marker_set_data"] = raw_markers_data
                process_time = time() - tic_process  # time to process all data

                for i in range(len(self.ports)):
                    try:
                        self.server_queue[i].get_nowait()
                    except Exception:
                        pass
                    self.server_queue[i].put_nowait(data_dic)
                if len(self.plots) != 0:
                    size = 1 if not self.plots_multiprocess else len(self.plots)
                    for i in range(size):
                        try:
                            self.plots_queue[i].get_nowait()
                        except Exception:
                            pass
                        self.plots_queue[i].put_nowait(data_dic)

                data_dic["absolute_time_frame"] = absolute_time_frame_dic
                data_dic["interface_latency"] = interface_latency
                data_dic["process_time"] = process_time
                data_dic["initial_time"] = initial_time
                data_dic["time_to_get_data"] = time_to_get_data

                # Save data
                if self.save_data is True:
                    tic_save = time()
                    data_dic["saving_time"] = saving_time
                    dic_to_save = dic_merger(data_dic, dic_to_save)
                    if save_count == int(self.stream_rate / self.save_frequency):
                        save(data_dic, self.save_path)
                        dic_to_save = {}
                        save_count = 0
                    save_count += 1
                    saving_time = time() - tic_save

                if tic - time() < 1 / self.stream_rate:
                    sleep(1 / self.stream_rate - (tic - time()))
                else:
                    print(
                        f"WARNING: Stream rate ({self.stream_rate}) is too high for the computer."
                        f"The actual stream rate is {1 / (tic - time())}"
                    )

    def stop(self):
        for process in self.processes:
            process.terminate()
            process.join()
