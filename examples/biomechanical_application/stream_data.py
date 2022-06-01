
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
from biosiglive.streaming.connection import Server
from biosiglive.io import save_data
from biosiglive.interfaces import pytrigno_interface, vicon_interface
from biosiglive.processing.data_processing import RealTimeProcessing
from biosiglive.processing.msk_functions import kalman_func
from biosiglive.gui.plot import LivePlot

vicon_package, biorbd = True, True

try:
    import biorbd
except ModuleNotFoundError:
    biorbd = False

try:
    from vicon_dssdk import ViconDataStream as VDS
except ModuleNotFoundError:
    vicon_package = False

Buff_size = 100000


class LiveData:
    def __init__(
            self,
            server_ip,
            server_ports=(),
            type=None,
            acquisition_rate=100,
            read_frequency=None,
            timeout=None,
            buff_size=Buff_size,
            stream_from="vicon",  # 'vicon' or 'pytrigno',
            device_host_ip="127.0.0.1",  # localhost
    ):

        # problem variables
        self.server_ip = server_ip
        self.server_ports = server_ports

        self.type = type if type is not None else "TCP"
        self.timeout = timeout if timeout else 10000
        self.read_frequency = read_frequency if read_frequency else acquisition_rate
        self.acquisition_rate = acquisition_rate
        self.emg_rate = None
        self.imu_rate = None
        self.markers_rate = None
        self.emg_windows = None
        self.markers_windows = None
        self.imu_windows = None
        self.recons_kalman = None
        self.model_path = None
        self.proc_emg = None
        self.proc_imu = None
        self.markers_dec = None
        self.emg_dec = None
        self.buff_size = buff_size
        self.save_data = save_data
        self.raw_data = False
        self.try_w_connection = True
        self.device_host_ip = device_host_ip
        self.offline_file_path = None
        self.smooth_markers = None
        # current_time = strftime("%Y%m%d-%H%M")
        # output_file = output_file if output_file else f"data_streaming_{current_time}"
        #
        # output_dir = output_dir if output_dir else "live_data"

        # if os.path.isdir(output_dir) is not True:
        #     os.mkdir(output_dir)
        #
        # if os.path.isfile(f"{output_dir}/{output_file}"):
        #     os.remove(f"{output_dir}/{output_file}")
        #
        # self.data_path = f"{output_dir}/{output_file}"

        # Init some variables
        self.plot_emg = ()
        self.mvc_list = ()
        self.norm_min_accel_value = ()
        self.norm_max_accel_value = ()
        self.norm_min_gyro_value = ()
        self.norm_max_gyro_value = ()
        self.norm_emg = None
        self.optim = False
        self.stream_emg, self.stream_markers, self.stream_imu, self.stream_force_plate = False, False, False, False
        self.subject_name, self.device_name = None, None
        self.iter = 0
        self.marker_names = ()
        self.nb_of_data_to_export = 1
        self.emg_empty, self.markers_empty = (), ()
        self.nb_emg, self.nb_marks = 0, 0
        # if not muscle_range:
        #     muscle_range = (0, 16)
        # self.nb_electrodes = muscle_range[1] - muscle_range[0] + 1
        # self.emg_sample = int(self.emg_rate / self.acquisition_rate)
        # self.imu_sample = int(self.imu_rate / self.acquisition_rate)
        # self.muscle_range = muscle_range if muscle_range else (0, 15)
        # self.imu_range = (self.muscle_range[0], self.muscle_range[0] + (self.nb_electrodes * 9))
        self.output_names = ()
        self.imu_output_names = ()
        self.emg_names = None
        self.imu_names = None
        if not self.offline_file_path:
            self.offline_time = 3

        # Multiprocess stuff
        manager = mp.Manager()
        self.server_queue = []
        for i in range(len(self.server_ports)):
            self.server_queue.append(manager.Queue())
        self.emg_queue_in = manager.Queue()
        self.emg_queue_out = manager.Queue()
        self.imu_queue_in = manager.Queue()
        self.imu_queue_out = manager.Queue()
        self.kin_queue_in = manager.Queue()
        self.kin_queue_out = manager.Queue()
        self.event_emg = mp.Event()
        self.event_kin = mp.Event()
        self.event_imu = mp.Event()
        self.event_vicon = mp.Event()
        self.process = mp.Process
        self.servers = []
        self.osc_clients = []
        self.count_server = 0
        self.count_osc = 0
        self.stream_from = stream_from
        if self.stream_from == 'vicon':
            self.interface = vicon_interface.ViconClient(ip=device_host_ip, port=801, init_now=False)
        elif self.stream_from == 'pytrigno':
            self.interface = pytrigno_interface.PytrignoClient(ip=device_host_ip)
        # elif not self.try_w_connection:
        #     self.interface = offline_interface()
        else:
            raise RuntimeError(f"{self.stream_from} is not implemented. Please stream data from vicon or pytrigno")

    def add_emg_device(self,
                       electrode_idx: Union[int, tuple],
                       device_name: Union[str, list] = None,
                       process: bool = True,
                       norm: bool = False,
                       mvc: list = None,
                       live_plot: bool = False,
                       rate: float = 2000):
        nb_emg = len(electrode_idx) if isinstance(electrode_idx, tuple) else 1
        if norm and not mvc:
            raise RuntimeError("Please provide mvc data to normalize emg signals or turn 'norm' to False")
        # else:
        #     if not isinstance(mvc, list):
        #         mvc = [mvc]
        #         if len(mvc) != nb_emg:
        #             raise RuntimeError("Number of MVC data have to be the same than number of electrodes. "
        #                                f"You have {len(mvc)} and {nb_emg}.")
        self.plot_emg = live_plot
        self.mvc_list = mvc
        self.stream_emg = True
        self.interface.add_device(name=device_name, range=electrode_idx, type="emg", rate=rate)
        self.interface.devices[-1].process_method = None if not process else self.interface.devices[-1].process_method

    def add_generic_device(self,
                       electrode_idx: Union[int, tuple],
                       names: Union[str, list] = None,
                       process: bool = False,
                       rate: float = 2000):
        self.stream_generic_device = True
        self.nb_emg = len(electrode_idx) if isinstance(electrode_idx, tuple) else 1
        self.interface.add_device(name=names, range=electrode_idx, type="generic_device", rate=rate, real_time=True)
        self.interface.devices[-1].process_method = None if not process else RealTimeProcessing().process_emg

    def add_imu_device(self,
                       electrode_idx: Union[int, tuple],
                       names: Union[str, list] = None,
                       process: bool = True,
                       norm_min_accel_value=None,
                       norm_max_accel_value=None,
                       norm_min_gyro_value=None,
                       norm_max_gyro_value=None,
                       rate: float = 148.1):
        self.stream_imu = True
        self.nb_imu = len(electrode_idx) if isinstance(electrode_idx, tuple) else 1
        self.interface.add_device(name=names, range=electrode_idx, type="imu", rate=rate)
        self.interface.devices[-1].process_method = None if not process else self.interface.devices[-1].process_method

    def add_force_plate_device(self):
        raise RuntimeError("Force plate not implemented yet.")

    def add_markers(self, nb_markers: int,
                    marker_set_name: str = None,
                    subject: str = None,
                    smooth_traj: bool = None,
                    rate: float = 100,
                    unlabeled: bool = False,
                    compute_kin: bool = False):
        self.stream_markers = True
        self.nb_marks = nb_markers
        self.smooth_markers = smooth_traj
        self.recons_kalman = compute_kin
        if self.stream_from == "pytrigno":
            raise RuntimeError("Impossible to stream markers data from pytrigno.")
        else:
            self.interface.add_markers(name=marker_set_name, rate=rate, unlabeled=unlabeled, subject_name=subject)

    def run(
            self,
            test_with_connection=True,
            save_data=True,
            output_file_path=None,
            offline_file_path=None,
            show_log=False,
    ):
        self.save_data = save_data
        self.output_file_path = output_file_path
        self.offline_file_path = offline_file_path
        self.optim = show_log
        self.try_w_connection = test_with_connection
        if self.try_w_connection is not True:
            print("[Warning] Debug mode without connection.")

        if self.offline_file_path:
            data_exp = sio.loadmat(self.offline_file_path)

        if self.try_w_connection is not True:
            if self.stream_imu:
                self.imu_exp = np.random.rand(self.nb_electrodes, 6, int(self.imu_rate * self.offline_time))
                self.imu_sample = int(self.imu_rate / self.acquisition_rate)
            if self.stream_emg:
                self.emg_sample = int(self.emg_rate / self.acquisition_rate)
                if self.offline_file_path and "emg" in data_exp.keys():
                    self.emg_exp = data_exp["emg"]
                    self.nb_electrodes = self.emg_exp.shape[0]
                else:
                    self.emg_exp = np.random.rand(self.nb_electrodes, int(self.emg_rate * self.offline_time))

            if self.stream_markers:
                if self.model_path:
                    if self.recons_kalman:
                        biomod = biorbd.Model(self.model_path)
                        self.nb_marks = biomod.nbMarkers()
                else:
                    self.nb_marks = self.nb_electrodes
                if self.offline_file_path and "markers" in data_exp.keys():
                    self.markers_exp = data_exp["markers"][:3, :, :]
                    self.nb_marks = self.markers_exp.shape[0]
                else:
                    self.markers_exp = np.random.rand(3, self.nb_marks, int(self.markers_rate * self.offline_time))

            self.init_frame = 0
            if not self.offline_file_path:
                self.last_frame = self.markers_rate * self.offline_time
            else:
                self.last_frame = min(self.markers_exp.shape[2], self.emg_exp.shape[1])
            self.m = self.init_frame
            self.c = self.init_frame * 20
            self.marker_names = []
        self.open_server()
        # processes = [self.process(name="reader", target=LiveData.save_streamed_data, args=(self,))]
        # for i in range(len(self.server_ports)):
        #     processes.append(self.process(name="listen" + f"_{i}", target=LiveData.open_server, args=(self,)))
        # if self.stream_emg:
        #     processes.append(self.process(name="process_emg", target=LiveData.emg_processing, args=(self,)))
        # if self.stream_imu:
        #     processes.append(self.process(name="process_imu", target=LiveData.imu_processing, args=(self,)))
        # if self.stream_markers:
        #     processes.append(self.process(name="kin", target=LiveData.recons_kin, args=(self,)))
        # for p in processes:
        #     p.start()
        #     if p.name.startswith("listen"):
        #         self.count_server += 1
        # for p in processes:
        #     p.join()

    def open_server(self):
        server = Server(self.server_ip, self.server_ports[self.count_server], type=self.type)
        server.start()
        data_queue = []
        while len(data_queue) == 0:
            try:
                data_queue = self.server_queue[self.count_server].get_nowait()
                is_working = True
            except mp.Queue.empty:
                is_working = False
                pass
            if is_working:
                server.client_listening(data_queue)

    def emg_processing(self):
        emg_tmp, emg_data = None, None
        c = 0
        emg_process_method = None
        for device in self.interface.devices:
            if device.type == 'emg':
                emg_process_method = device.get_process_method()
        while True:
            try:
                emg_data = self.emg_queue_in.get_nowait()
                is_working = True
            except mp.Queue.empty:
                is_working = False

            if is_working:
                if self.try_w_connection is not True:
                    if c < self.emg_exp.shape[1]:
                        emg_tmp = self.emg_exp[: self.nb_electrodes, c: c + self.emg_sample]
                        c += self.emg_sample
                    else:
                        c = 0
                else:
                    emg_tmp = emg_data["emg_tmp"]
                raw_emg, emg_proc = emg_data["raw_emg"], emg_data["emg_proc"]
                raw_emg, emg_proc = emg_process_method(raw_emg,
                                                       emg_proc,
                                                       emg_tmp,
                                                       mvc_list=self.mvc_list,
                                                       norm_emg=self.norm_emg
                                                       )

                self.emg_queue_out.put({"raw_emg": raw_emg, "emg_proc": emg_proc})
                self.event_emg.set()

    def imu_processing(self):
        imu_tmp, imu_data = None, None
        d = 0
        imu_process_method = None
        for device in self.interface.devices:
            if device.type == 'imu':
                imu_process_method = device.get_process_method()
        while True:
            try:
                imu_data = self.imu_queue_in.get_nowait()
                is_working = True
            except:
                is_working = False

            if is_working:
                if self.try_w_connection is not True:
                    if d < self.IM_exp.shape[2]:
                        imu_tmp = self.IM_exp[: self.nb_electrodes, :, d: d + self.imu_sample]
                        d += self.imu_sample
                    else:
                        d = 0
                else:
                    imu_tmp = imu_data["imu_tmp"]

                accel_tmp = imu_tmp[:, :3, :]
                gyro_tmp = imu_tmp[:, 3:6, :]
                if self.stream_from == "vicon":
                    # convert rad/s into deg/s when vicon is used
                    gyro_tmp = gyro_tmp * (180 / np.pi)

                if self.stream_from == "pytrigno":
                    # convert data from G into m/s2 when pytrigno is used
                    accel_tmp = accel_tmp * 9.81

                raw_imu, imu_proc = imu_data["raw_imu"], imu_data["imu_proc"]
                if len(raw_imu) != 0:
                    if len(imu_proc.shape) == 3:
                        raw_accel, accel_proc = (
                            raw_imu[: self.nb_electrodes, :3, :],
                            imu_proc[: self.nb_electrodes, :3, :],
                        )
                        raw_gyro, gyro_proc = (
                            raw_imu[: self.nb_electrodes, 3:6, :],
                            imu_proc[: self.nb_electrodes, 3:6, :],
                        )
                    else:
                        raw_accel, accel_proc = raw_imu[: self.nb_electrodes, :3, :], imu_proc[: self.nb_electrodes, :]
                        raw_gyro, gyro_proc = raw_imu[: self.nb_electrodes, 3:6, :], imu_proc[self.nb_electrodes:, :]
                else:
                    raw_accel, accel_proc = raw_imu, imu_proc
                    raw_gyro, gyro_proc = raw_imu, imu_proc

                raw_accel, accel_proc = imu_process_method(
                    accel_proc,
                    raw_accel,
                    accel_tmp,
                    self.imu_windows,
                    self.imu_sample,
                    ma_win=30,
                    accel=True,
                    norm_min_bound=self.norm_min_accel_value,
                    norm_max_bound=self.norm_max_accel_value,
                    squared=False,
                )
                raw_gyro, gyro_proc = imu_process_method(
                    gyro_proc,
                    raw_gyro,
                    gyro_tmp,
                    self.imu_windows,
                    self.imu_sample,
                    ma_win=30,
                    norm_min_bound=self.norm_min_gyro_value,
                    norm_max_bound=self.norm_max_gyro_value,
                    squared=False,
                )
                if len(accel_proc.shape) == 3:
                    raw_imu, imu_proc = (
                        np.concatenate((raw_accel, raw_gyro), axis=1),
                        np.concatenate((accel_proc, gyro_proc), axis=1),
                    )
                else:
                    raw_imu, imu_proc = (
                        np.concatenate((raw_accel, raw_gyro), axis=1),
                        np.concatenate((accel_proc, gyro_proc), axis=0),
                    )
                self.imu_queue_out.put({"raw_imu": raw_imu, "imu_proc": imu_proc})
                self.event_imu.set()

    def recons_kin(self):
        model, kalman, markers_data = None, None, None
        if self.recons_kalman:
            model = biorbd.Model(self.model_path)
            freq = 100  # Hz
            params = biorbd.KalmanParam(freq)
            kalman = biorbd.KalmanReconsMarkers(model, params)

        while True:
            try:
                markers_data = self.kin_queue_in.get_nowait()
                is_working = True
            except mp.Queue.empty:
                is_working = False

            if is_working:
                markers = markers_data["markers"]
                states = markers_data["states"]
                if self.try_w_connection:
                    markers_tmp = markers_data["markers_tmp"]
                else:
                    markers_tmp = self.markers_exp[:, :, self.m: self.m + 1]
                    self.m = self.m + 1 if self.m < self.last_frame else self.init_frame

                markers_tmp = markers_tmp * 0.001
                if self.recons_kalman:
                    states_tmp = kalman_func(markers_tmp, model, return_q_dot=False, kalman=kalman)

                    if len(states) == 0:
                        states = states_tmp
                    else:
                        if states.shape[1] < self.markers_windows:
                            states = np.append(states, states_tmp, axis=1)
                        else:
                            states = np.append(states[:, 1:], states_tmp, axis=1)

                if len(markers) != 0:
                    if self.smooth_markers:
                        # if self.recons_kalman:
                        #     markers_from_kalman = np.array(
                        #         [mark.to_array() for mark in model.markers(states[:model.nbQ(), -1])]).T
                        #     for i in range(markers_tmp.shape[1]):
                        #         if np.product(markers_tmp[:, i, :]) == 0:
                        #             markers_tmp[:, i, 0] = markers_from_kalman[:, i]
                        # else:
                        for i in range(markers_tmp.shape[1]):
                            if np.product(markers_tmp[:, i, :]) == 0:
                                markers_tmp[:, i, :] = markers[:, i, -1:]
                if len(markers) == 0:
                    markers = markers_tmp
                else:
                    if markers.shape[2] < self.markers_windows:
                        markers = np.append(markers, markers_tmp, axis=2)
                    else:
                        markers = np.append(markers[:, :, 1:], markers_tmp[:, :, -1:], axis=2)
                self.kin_queue_out.put({"states": states, "markers": markers})
                self.event_kin.set()

    # def init_stream(self):
    #     if self.device == "vicon":
    #         vicon_interface.ViconClient(self.device_host_ip, 801)
    #
    #     elif self.device == "pytrigno":
    #         pytrigno_interface.PytrignoClient(self.device_host_ip)

    def init_live_plot(self, multi=True, names=None):
        """
        Initialize the live plot.

        Parameters
        ----------
        multi: bool
            If True, the live plot is initialized for multi-threads plot.

        Returns
        -------
        rplt: list of live plot, layout: qt layout, qt app : pyqtapp, checkbox : list of checkbox

        """
        self.plot_app = LivePlot(multi_process=multi)
        self.plot_app.add_new_plot("EMG", "curve", names)
        rplt, layout, app, box = self.plot_app.init_plot_window(self.plot_app.plot[0],
                                                                use_checkbox=True,
                                                                remote=True
                                                                )
        return rplt, layout, app, box

    def save_streamed_data(self):

        emg_dec = self.emg_dec
        markers_dec = self.markers_dec
        raw_emg = []
        raw_imu = []
        imu_proc = []
        emg_proc = []
        markers = []
        states = []
        app, rplt, box = None, None, None
        vicon_latency_total = 0
        initial_time = 0
        absolute_time_frame = 0
        if self.try_w_connection:
            self.interface.init_client()
        self.nb_marks = len(self.marker_names)
        if self.plot_emg:
            rplt, win_emg, app, box = self.init_live_plot(multi=True, names=None)
        delta = 0
        delta_tmp = 0
        self.iter = 0
        dic_to_put = {}
        while True:
            tic = time()
            if self.iter == 0:
                initial_time = time() - tic
                print("Data start streaming")
            self.interface.get_frame()
            if self.try_w_connection:
                frame = self.interface.get_frame()
                stream_latency = self.interface.get_latency()
            else:
                frame = True
            absolute_time_frame = datetime.datetime.now()

            absolute_time_frame_dic = {"day": absolute_time_frame.day,
                                       "hour": absolute_time_frame.hour,
                                       "hour_s": absolute_time_frame.hour * 3600,
                                       "minute": absolute_time_frame.minute,
                                       "minute_s": absolute_time_frame.minute * 60,
                                       "second": absolute_time_frame.second,
                                       "millisecond": int(absolute_time_frame.microsecond / 1000),
                                       "millisecond_s": int(absolute_time_frame.microsecond / 1000) * 0.001,
                                       }

            if frame:
                if self.stream_emg or self.stream_generic_device or self.stream_imu:
                    all_device_data = self.interface.get_device_data("all")
                    for i, device in enumerate(self.interface.devices):
                        if device.type == "emg":
                            emg_tmp, emg_names = all_device_data[i]
                            self.emg_queue_in.put_nowait({"raw_emg": raw_emg, "emg_proc": emg_proc, "emg_tmp": emg_tmp})

                        elif device.type == "imu":
                            imu_tmp, _ = all_device_data[i]
                            self.imu_queue_in.put_nowait({"raw_imu": raw_imu, "imu_proc": imu_proc, "imu_tmp": imu_tmp})

                        elif device.type == "generic_device":
                            generic_device_tmp, _ = all_device_data[i]

                if self.stream_markers:
                    all_markers_tmp, all_occluded = self.interface.get_markers_data()
                    markers_tmp = all_markers_tmp[0]
                    occluded = all_occluded[0]
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
                    dic_to_put["emg_names"] = emg_names
                    dic_to_put["raw_emg"] = np.around(raw_emg, decimals=emg_dec)
                    dic_to_put["emg_proc"] = np.around(emg_proc, decimals=emg_dec)

                if self.stream_markers:
                    self.event_kin.wait()
                    kin = self.kin_queue_out.get_nowait()
                    self.event_kin.clear()
                    states, markers = kin["states"], kin["markers"]
                    dic_to_put["markers"] = np.around(markers, decimals=markers_dec)
                    dic_to_put["kalman"] = states
                    dic_to_put["marker_names"] = self.marker_names

                if self.stream_imu:
                    self.event_imu.wait()
                    imu = self.imu_queue_out.get_nowait()
                    self.event_imu.clear()
                    raw_imu, imu_proc = imu["raw_imu"], imu["imu_proc"]
                    # dic_to_put["imu_names"] = imu_names
                    dic_to_put["raw_imu"] = raw_imu
                    dic_to_put["imu_proc"] = imu_proc

            dic_to_put["acquisition_rate"] = self.acquisition_rate
            dic_to_put["absolute_time_frame"] = absolute_time_frame_dic
            if self.device == "vicon":
                dic_to_put["vicon_latency"] = vicon_latency_total
            process_time = time() - tic  # time to process all data + time to get data
            for i in range(len(self.server_ports)):
                try:
                    self.server_queue[i].get_nowait()
                except mp.Queue().empty():
                    pass
                self.server_queue[i].put_nowait(dic_to_put)

            if self.plot_emg:
                self.plot_app.update_plot_window(self.plot_app.plot[0], raw_emg, app, rplt, box)

            self.iter += 1

            # Save data
            if self.save_data is True:
                data_to_save = {
                    "process_delay": process_time,
                    "absolute_time_frame": absolute_time_frame_dic,
                    "vicon_latency_total": vicon_latency_total,
                    "initial_time": initial_time,
                    "emg_freq": self.emg_rate,
                    "IM_freq": self.imu_rate,
                    "acquisition_freq": self.acquisition_rate,
                }
                if self.stream_emg:
                    data_to_save["emg_proc"] = emg_proc[:, -1:]
                    data_to_save["raw_emg"] = raw_emg[:, -self.emg_sample:]

                if self.stream_markers:
                    data_to_save["markers"] = markers[:3, :, -1:]

                if self.recons_kalman:
                    data_to_save["kalman"] = states[:, -1:]

                if self.stream_imu:
                    if imu_proc.shape == 3:
                        data_to_save["accel_proc"] = imu_proc[:, 0:3, -1:]
                        data_to_save["raw_accel"] = raw_imu[:, 0:3, -self.imu_sample:]
                        data_to_save["gyro_proc"] = imu_proc[:, 3:6, -1:]
                        data_to_save["raw_gyro"] = raw_imu[:, 3:6, -self.imu_sample:]
                    else:
                        data_to_save["accel_proc"] = imu_proc[: self.nb_electrodes, -1:]
                        data_to_save["raw_accel"] = raw_imu[:, 0:3, -self.imu_sample:]
                        data_to_save["gyro_proc"] = imu_proc[self.nb_electrodes:, -1:]
                        data_to_save["raw_gyro"] = raw_imu[:, 3:6, -self.imu_sample:]

                save_data.add_data_to_pickle(data_to_save, self.data_path)

            duration = time() - tic
            if 1 / duration > self.acquisition_rate:
                sleep((1 / self.acquisition_rate) - duration)
            # delta, delta_tmp = self._loop_sleep(delta_tmp, delta, tic)


if __name__ == '__main__':
    server_ip = "127.0.0.1"
    server_ports = [50000]
    live_stream = LiveData(server_ip=server_ip,
                           server_ports=server_ports,
                           stream_from="vicon",
                           device_host_ip="127.0.0.1",
                           acquisition_rate=100)

    live_stream.add_emg_device(electrode_idx=(0, 9),
                               device_name="EMG",
                               process=True,
                               norm=False,
                               live_plot=False,
                               rate=2000)
    emg_processing = RealTimeProcessing()
    emg_processing.ma_win = 2000
    live_stream.interface.devices[-1].set_process_method(emg_processing.process_emg)

    live_stream.add_imu_device(electrode_idx=(0,9),
                               names='IMU',
                               process=True)

    live_stream.add_markers(nb_markers=15,
                            subject="subject_1",
                            smooth_traj=False,
                            rate=100,
                            unlabeled=False,
                            compute_kin=True
                            )
