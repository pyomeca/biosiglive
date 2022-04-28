import socket
import struct
try:
    from pythonosc.udp_client import SimpleUDPClient
    osc_package = True
except ModuleNotFoundError:
    osc_package = False

try:
    import pytrigno
except ModuleNotFoundError:
    pass

import sys
from time import time, sleep, strftime
import datetime
import scipy.io as sio
import numpy as np
from math import ceil
from biosiglive.data_plot import init_plot_emg, update_plot_emg
from biosiglive.data_processing import process_emg_rt, process_imu, add_data_to_pickle
import multiprocessing as mp
import os
import json

vicon_package, biorbd_eigen = True, True

try:
    import biorbd
except ModuleNotFoundError:
    biorbd = False

try:
    from vicon_dssdk import ViconDataStream as VDS
except ModuleNotFoundError:
    vicon_package = False

Buff_size = 100000


class Server:
    def __init__(
        self,
        IP,
        server_ports=[],
        osc_ports=[],
        type=None,
        acquisition_rate=100,
        emg_rate=2000,
        imu_rate=148.1,
        markers_rate=100,
        emg_windows=2000,
        markers_windows=100,
        imu_windows=100,
        read_frequency=None,
        recons_kalman=False,
        model_path=None,
        proc_emg=True,
        proc_imu=True,
        markers_dec=4,
        emg_dec=10,
        timeout=None,
        buff_size=Buff_size,
        device="vicon",  # 'vicon' or 'pytrigno',
        device_host_ip="127.0.0.1",  # localhost
        muscle_range=None,
        output_file=None,
        output_dir=None,
        save_data=True,
        offline_file_path=None,
        smooth_markers=True
    ):

        # problem variables
        self.IP = IP
        if isinstance(server_ports, list):
            self.ports = server_ports
        else:
            self.ports = [server_ports]
        if isinstance(osc_ports, list):
            self.osc_ports = osc_ports
        else:
            self.osc_ports = [osc_ports]
        if len(self.osc_ports) == 0 and len(self.ports) == 0:
            raise RuntimeError("Please define at least one port for either osc streaming or server streaming.")
        
        self.type = type if type is not None else "TCP"
        self.timeout = timeout if timeout else 10000
        self.read_frequency = read_frequency if read_frequency else acquisition_rate
        self.acquisition_rate = acquisition_rate
        self.emg_rate = emg_rate
        self.imu_rate = imu_rate
        self.markers_rate = markers_rate
        self.emg_windows = emg_windows
        self.markers_windows = markers_windows
        self.imu_windows = imu_windows
        self.recons_kalman = recons_kalman
        self.model_path = model_path
        self.proc_emg = proc_emg
        self.proc_imu = proc_imu
        self.markers_dec = markers_dec
        self.emg_dec = emg_dec
        self.buff_size = buff_size
        self.save_data = save_data
        self.raw_data = False
        self.try_w_connection = True
        self.device = device
        self.device_host_ip = device_host_ip
        self.offline_file_path = offline_file_path
        self.smooth_markers = smooth_markers
        current_time = strftime("%Y%m%d-%H%M")
        output_file = output_file if output_file else f"data_streaming_{current_time}"

        output_dir = output_dir if output_dir else "live_data"

        if os.path.isdir(output_dir) is not True:
            os.mkdir(output_dir)

        if os.path.isfile(f"{output_dir}/{output_file}"):
            os.remove(f"{output_dir}/{output_file}")

        self.data_path = f"{output_dir}/{output_file}"

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
        if not muscle_range:
            muscle_range = (0, 16)
        self.nb_electrodes = muscle_range[1] - muscle_range[0] + 1
        self.emg_sample = int(self.emg_rate / self.acquisition_rate)
        self.imu_sample = int(self.imu_rate / self.acquisition_rate)
        self.muscle_range = muscle_range if muscle_range else (0, 15)
        self.imu_range = (self.muscle_range[0], self.muscle_range[0] + (self.nb_electrodes * 9))
        self.output_names = ()
        self.imu_output_names = ()
        self.emg_names = None
        self.imu_names = None
        if not self.offline_file_path:
            self.offline_time = 3

        # Multiprocess stuff
        manager = mp.Manager()
        self.server_queue = []
        for i in range(len(self.ports) + len(self.osc_ports)):
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

    @staticmethod
    def __server_sock(type):
        if type == "TCP" or type is None:
            return socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        elif type == "UDP":
            return socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def run(
        self,
        stream_emg=True,
        stream_markers=True,
        stream_imu=False,
        stream_force_plate=False,
        stream_generic_device=False,
        norm_emg=True,
        optim=False,
        plot_emg=False,
        mvc_list=None,
        norm_min_accel_value=None,
        norm_max_accel_value=None,
        norm_min_gyro_value=None,
        norm_max_gyro_value=None,
        subject_name=None,
        emg_device_name=None,
        imu_device_name=None,
        test_with_connection=True,
        generic_device_name=None,
    ):

        self.device_name = [emg_device_name]
        if not isinstance(generic_device_name, list):
            generic_device_name = [generic_device_name]
        for i in generic_device_name:
            self.device_name.append(i)
        self.imu_device_name = imu_device_name
        self.plot_emg = plot_emg
        self.mvc_list = mvc_list
        self.norm_emg = norm_emg
        shape_mvc = None
        if self.norm_emg is True and self.mvc_list is None:
            raise RuntimeError("Please define a mvc list to normalize emg signals or turn 'norm_emg' to False")
        if self.norm_emg and self.mvc_list is not None:
            if isinstance(mvc_list, list):
                if len(self.mvc_list) != self.nb_electrodes:
                    shape_mvc = len(self.mvc_list)
            else:
                if self.mvc_list.shape[0] != self.nb_electrodes:
                    shape_mvc = self.mvc_list.shape[0]
            if shape_mvc:
                raise RuntimeError("Number of MVC data have to be the same than number of electrodes. "
                                   f"You have {shape_mvc} and {self.nb_electrodes}.")
        self.optim = optim
        self.stream_emg = stream_emg
        self.stream_markers = stream_markers
        self.stream_imu = stream_imu
        self.stream_generic_device = stream_generic_device
        self.stream_force_plate = stream_force_plate
        self.subject_name = subject_name
        self.norm_min_accel_value = norm_min_accel_value
        self.norm_max_accel_value = norm_max_accel_value
        self.norm_max_gyro_value = norm_max_gyro_value
        self.norm_min_gyro_value = norm_min_gyro_value
        self.try_w_connection = test_with_connection
        if self.try_w_connection is not True:
            print("[Warning] Debug mode without main.")

        if self.offline_file_path:
            data_exp = sio.loadmat(self.offline_file_path)

        data_type = []
        if self.stream_emg:
            data_type.append("emg")
        if self.stream_markers or self.recons_kalman:
            data_type.append("markers")
        if self.stream_imu:
            data_type.append("imu")
        if self.stream_generic_device:
            data_type.append("generic_device")
        if self.stream_force_plate:
            raise RuntimeError("Not implemented yet")

        self.imu_sample = int(self.imu_rate / self.acquisition_rate)
        if self.try_w_connection is not True:
            if self.stream_imu:
                self.IM_exp = np.random.rand(self.nb_electrodes, 6, int(self.imu_rate * self.offline_time))
                self.imu_sample = int(self.imu_rate / self.acquisition_rate)
            if self.stream_emg:
                self.emg_sample = int(self.emg_rate / self.acquisition_rate)
                if self.offline_file_path and "emg" in data_exp.keys():
                    self.emg_exp = data_exp["emg"]
                    self.nb_electrodes = self.emg_exp.shape[0]
                else:
                    self.emg_exp = np.random.rand(self.nb_electrodes,  int(self.emg_rate * self.offline_time))

            if stream_markers:
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

        # Start connexion
        for i in range(len(self.ports)):
            if self.type == "TCP":
                self.servers.append(socket.socket(socket.AF_INET, socket.SOCK_STREAM))
            elif self.type == "UDP":
                self.servers.append(socket.socket(socket.AF_INET, socket.SOCK_DGRAM))
            else:
                raise RuntimeError(f"Invalid type of connexion ({type}). Type must be 'TCP' or 'UDP'.")
            try:
                self.servers[i].setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.servers[i].bind((self.IP, self.ports[i]))
                if self.type != "UDP":
                    self.servers[i].listen(10)
                    self.inputs = [self.servers[i]]
                    self.outputs = []
                    self.message_queues = {}

            except ConnectionError:
                raise RuntimeError("Unknown error. Server is not listening.")

        for i in range(len(self.osc_ports)):
            try:
                self.osc_clients.append(SimpleUDPClient(self.IP, self.osc_ports[i]))
                print(f"Streaming OSC {i} activated on '{self.IP}:{self.osc_ports[i]}")
            except ConnectionError:
                raise RuntimeError("Unknown error. OSC client not open.")

        if self.try_w_connection:
            if self.device == "pytrigno":
                self._init_pytrigno()

        processes = [self.process(name="reader", target=Server.save_streamed_data, args=(self,))]
        for i in range(len(self.ports)):
            processes.append(self.process(name="listen" + f"_{i}", target=Server.open_server, args=(self,)))
        for i in range(len(self.osc_ports)):
            processes.append(self.process(name="osc_listen" + f"_{i}", target=Server.open_server, args=(self, True)))
        if self.stream_emg:
            processes.append(self.process(name="process_emg", target=Server.emg_processing, args=(self,)))
        if self.stream_imu:
            processes.append(self.process(name="process_imu", target=Server.imu_processing, args=(self,)))
        if self.stream_markers:
            processes.append(self.process(name="kin", target=Server.recons_kin, args=(self,)))
        for p in processes:
            p.start()
            if p.name.startswith("listen"):
                self.count_server += 1
            if p.name.startswith("osc"):
                self.count_osc += 1
        for p in processes:
            p.join()

    def open_server(self, osc_type=False):
        server_idx = 0
        osc_idx = 0
        if not osc_type:
            server_idx = self.count_server
            print(
                f"{self.type} server {server_idx} is listening on '{self.IP}:{self.ports[server_idx]}' "
                f"and waiting for a client."
            )
        else:
            osc_idx = self.count_osc

        if osc_type:
            server_idx = osc_idx
        while 1:
            if not osc_type:
                connection, ad = self.servers[server_idx].accept()
                if self.optim is not True:
                    print(f"new main from {ad}")
                if self.type == "TCP":
                    message = json.loads(connection.recv(self.buff_size))
                    if self.optim is not True:
                        print(f"client sended {message}")
            data_queue = {}

            while len(data_queue) == 0:
                try:
                    data_queue = self.server_queue[server_idx].get_nowait()
                    is_working = True
                except:
                    is_working = False
                    pass
                if is_working:
                    if not osc_type:
                        for key in message:
                            self.__dict__[key] = message[key]
                        self.acquisition_rate = data_queue["acquisition_rate"]
                        absolute_time_frame = data_queue["absolute_time_frame"]
                        norm_emg = message["norm_emg"]
                        mvc_list = message["mvc_list"]
                        self.nb_of_data_to_export = message["nb_of_data_to_export"] if message["nb_of_data_to_export"] else 1
                        if self.acquisition_rate < self.read_frequency:
                            ratio = 1
                        else:
                            ratio = int(self.acquisition_rate / self.read_frequency)
                        data_to_prepare = {}

                        if len(message["command"]) != 0:
                            for i in message["command"]:
                                if i == "emg":
                                    if self.stream_emg:
                                        if self.raw_data:
                                            raw_emg = data_queue["raw_emg"]
                                            data_to_prepare["raw_emg"] = raw_emg
                                        emg = data_queue["emg_proc"]
                                        if norm_emg:
                                            if isinstance(mvc_list, np.ndarray) is True:
                                                if len(mvc_list.shape) == 1:
                                                    quot = mvc_list.reshape(-1, 1)
                                                else:
                                                    quot = mvc_list
                                            else:
                                                quot = np.array(mvc_list).reshape(-1, 1)
                                        else:
                                            quot = [1]
                                        data_to_prepare["emg"] = emg / quot
                                    else:
                                        raise RuntimeError(f"Data you asking for ({i}) is not streaming")
                                elif i == "markers":
                                    if self.stream_markers:
                                        markers = data_queue["markers"]
                                        data_to_prepare["markers"] = markers
                                    else:
                                        raise RuntimeError(f"Data you asking for ({i}) is not streaming")

                                elif i == "imu":
                                    if self.stream_imu:
                                        if self.raw_data:
                                            raw_imu = data_queue["raw_imu"]
                                            data_to_prepare["raw_imu"] = raw_imu
                                        imu = data_queue["imu_proc"]
                                        data_to_prepare["imu"] = imu
                                    else:
                                        raise RuntimeError(f"Data you asking for ({i}) is not streaming")

                                elif i == "force plate":
                                    raise RuntimeError("Not implemented yet.")
                                else:
                                    raise RuntimeError(
                                        f"Unknown command '{i}'. Command must be :'emg', 'markers' or 'imu' "
                                    )

                        if message["kalman"] is True:
                            if self.recons_kalman:
                                angle = data_queue["kalman"]
                                data_to_prepare["kalman"] = angle
                            else:
                                raise RuntimeError(
                                    f"Kalman reconstruction is not activate. "
                                    f"Please turn server flag recons_kalman to True."
                                )

                        # prepare data
                        dic_to_send = self.prepare_data(data_to_prepare, ratio)

                        if message["get_names"] is True:
                            dic_to_send["marker_names"] = data_queue["marker_names"]
                            dic_to_send["emg_names"] = data_queue["emg_names"]
                        dic_to_send["absolute_time_frame"] = absolute_time_frame
                        if self.optim is not True:
                            print("Sending data to client...")
                            print(f"data sended : {dic_to_send}")
                        print(np.array(dic_to_send["raw_emg"]).shape)
                        encoded_data = json.dumps(dic_to_send).encode()
                        encoded_data = struct.pack('>I', len(encoded_data)) + encoded_data
                        try:
                            connection.sendall(encoded_data)
                        except:
                            pass

                        if self.optim is not True:
                            print(f"Data of size {sys.getsizeof(dic_to_send)} sent to the client.")

                    elif osc_type:
                        if self.stream_emg:
                            emg_proc = np.array(data_queue["emg_proc"])[:, -1:]
                            emg_proc = emg_proc.reshape(emg_proc.shape[0])
                            self.osc_clients[osc_idx].send_message("/emg/processed/", emg_proc.tolist())
                        if self.stream_imu:
                            imu = np.array(data_queue["imu_proc"])[:, :, -1:]
                            accel_proc = imu[:, :3, :]
                            accel_proc = accel_proc.reshape(accel_proc.shape[0])
                            gyro_proc = imu[:, 3:6, :]
                            gyro_proc = gyro_proc.reshape(gyro_proc.shape[0])
                            self.osc_clients[osc_idx].send_message("/imu/", imu.tolist())
                            self.osc_clients[osc_idx].send_message("/accel/", accel_proc.tolist())
                            self.osc_clients[osc_idx].send_message("/gyro/", gyro_proc.tolist())

    def prepare_data(self, data_to_prep, ratio):
        """
        Prepare data to send to the client.
        Parameters
        ----------
        data_to_prep : dict
            Data to prepare.
        ratio : int
            Ratio of data to send.
        Returns
        -------
        dict
            Data prepared.
        """

        for key in data_to_prep.keys():
            nb_of_data_to_export = self.nb_of_data_to_export
            if len(data_to_prep[key].shape) == 2:
                if self.raw_data is not True or key != "raw_emg":
                    data_to_prep[key] = data_to_prep[key][:, ::ratio]
                if self.raw_data is True and key == "raw_emg":
                    nb_of_data_to_export = self.emg_sample * nb_of_data_to_export
                data_to_prep[key] = data_to_prep[key][:, -nb_of_data_to_export:].tolist()
            elif len(data_to_prep[key].shape) == 3:
                if self.raw_data is not True or key != "raw_imu":
                    data_to_prep[key] = data_to_prep[key][:, :, ::ratio]
                if self.raw_data is True and key == "raw_imu":
                    nb_of_data_to_export = self.imu_sample * nb_of_data_to_export
                data_to_prep[key] = data_to_prep[key][:, :, -nb_of_data_to_export:].tolist()
        return data_to_prep

    def _init_pytrigno(self):
        if self.stream_emg:
            self.emg_sample = int(self.emg_rate / self.acquisition_rate)
            if self.norm_emg is True and len(self.mvc_list) != self.nb_electrodes:
                raise RuntimeError(
                    f"Length of the mvc list ({self.mvc_list}) "
                    f"not consistent with emg number ({self.nb_electrodes})."
                )
            self.dev_emg = pytrigno.TrignoEMG(
                channel_range=self.muscle_range, samples_per_read=self.emg_sample, host=self.device_host_ip
            )
            self.dev_emg.start()

        if self.stream_imu:
            self.imu_sample = int(self.imu_rate / self.acquisition_rate)

            self.dev_imu = pytrigno.TrignoIM(
                channel_range=self.imu_range, samples_per_read=self.imu_sample, host=self.device_host_ip
            )
            self.dev_imu.start()

    def _init_vicon_client(self):
        address = f"{self.device_host_ip}:801"
        print(f"Connection to ViconDataStreamSDK at : {address} ...")
        self.vicon_client = VDS.Client()
        self.vicon_client.Connect(address)
        self.vicon_client.EnableSegmentData()
        self.vicon_client.EnableDeviceData()
        self.vicon_client.EnableMarkerData()
        self.vicon_client.EnableUnlabeledMarkerData()

        a = self.vicon_client.GetFrame()
        while a is not True:
            a = self.vicon_client.GetFrame()

        acquisition_rate = self.vicon_client.GetFrameRate()
        if acquisition_rate != self.acquisition_rate:
            print(
                f"[WARNING] Vicon system rate ({acquisition_rate} Hz) is different than system rate you chosen "
                f"({self.acquisition_rate} Hz). System rate is now set to : {acquisition_rate} Hz."
            )
            self.acquisition_rate = acquisition_rate

        if self.stream_emg:
            self.device_name = self.device_name if self.device_name else self.vicon_client.GetDeviceNames()[2][0]
            self.device_info = self.vicon_client.GetDeviceOutputDetails(self.device_name)
            self.emg_sample = ceil(self.emg_rate / self.acquisition_rate)
            self.emg_empty = np.zeros((len(self.device_info), self.emg_sample))
            # self.output_names, self.emg_names = self.get_emg(init=True)
            # self.nb_emg = len(self.output_names)
            if self.norm_emg is True and len(self.mvc_list) != self.nb_electrodes:
                raise RuntimeError(
                    f"Length of the mvc list ({self.mvc_list}) "
                    f"not consistent with emg number ({self.nb_electrodes})."
                )

        if self.stream_imu:
            self.imu_device_name = (
                self.imu_device_name if self.imu_device_name else self.vicon_client.GetDeviceNames()[3][0]
            )
            self.imu_device_info = self.vicon_client.GetDeviceOutputDetails(self.imu_device_name)
            self.imu_sample = ceil(self.imu_rate / self.acquisition_rate)
            self.imu_empty = np.zeros((144, self.imu_sample))
            # self.imu_output_names, self.imu_names = self.get_imu(init=True)
            # self.nb_imu = len(self.imu_output_names)

        if self.stream_markers:
            self.subject_name = self.subject_name if self.subject_name else self.vicon_client.GetSubjectNames()[0]
            self.vicon_client.EnableMarkerData()
            self.vicon_client.EnableUnlabeledMarkerData()
            self.vicon_client.EnableMarkerRayData()
            self.marker_names = self.vicon_client.GetMarkerNames(self.subject_name)
            self.markers_empty = np.ndarray((3, len(self.marker_names), 1))

    def _loop_sleep(self, delta_tmp, delta, tic, print_time=False):
        delta = (delta_tmp + delta) / 2
        toc = time() - tic
        time_to_sleep = (1 / self.acquisition_rate) - toc - delta
        if time_to_sleep > 0:
            sleep(time_to_sleep)
        loop_duration = time() - tic
        delta_tmp = loop_duration - 1 / self.acquisition_rate
        if print_time is True:
            toc = time() - tic
            print(toc)
        return delta, delta_tmp

    def emg_processing(self):
        # self.event_vicon.wait()
        c = 0
        while True:
            try:
                emg_data = self.emg_queue_in.get_nowait()
                is_working = True
            except:
                is_working = False
                pass

            if is_working:
                if self.try_w_connection is not True:
                    if c < self.emg_exp.shape[1]:
                        emg_tmp = self.emg_exp[: self.nb_electrodes, c : c + self.emg_sample]
                        c += self.emg_sample
                    else:
                        c = 0
                else:
                    emg_tmp = emg_data["emg_tmp"]
                ma_win = 200
                raw_emg, emg_proc = emg_data["raw_emg"], emg_data["emg_proc"]
                raw_emg, emg_proc = process_emg_rt(
                    raw_emg,
                    emg_proc,
                    emg_tmp,
                    mvc_list=self.mvc_list,
                    ma_win=ma_win,
                    emg_win=self.emg_windows,
                    emg_freq=self.emg_rate,
                    norm_emg=self.norm_emg,
                    lpf=False,
                )
                self.emg_queue_out.put({"raw_emg": raw_emg, "emg_proc": emg_proc})
                self.event_emg.set()

    def imu_processing(self):
        d = 0
        while True:
            try:
                imu_data = self.imu_queue_in.get_nowait()
                is_working = True
            except:
                is_working = False

            if is_working:
                if self.try_w_connection is not True:
                    if d < self.IM_exp.shape[2]:
                        imu_tmp = self.IM_exp[: self.nb_electrodes, :, d : d + self.imu_sample]
                        d += self.imu_sample
                    else:
                        d = 0
                else:
                    imu_tmp = imu_data["imu_tmp"]

                accel_tmp = imu_tmp[:, :3, :]
                gyro_tmp = imu_tmp[:, 3:6, :]
                if self.device == "vicon":
                    # convert rad/s into deg/s when vicon is used
                    gyro_tmp = gyro_tmp * (180 / np.pi)

                if self.device == "pytrigno":
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
                        raw_gyro, gyro_proc = raw_imu[: self.nb_electrodes, 3:6, :], imu_proc[self.nb_electrodes :, :]
                else:
                    raw_accel, accel_proc = raw_imu, imu_proc
                    raw_gyro, gyro_proc = raw_imu, imu_proc

                raw_accel, accel_proc = process_imu(
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
                raw_gyro, gyro_proc = process_imu(
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
        if self.recons_kalman:
            model = biorbd.Model(self.model_path)
            freq = 100  # Hz
            params = biorbd.KalmanParam(freq)
            kalman = biorbd.KalmanReconsMarkers(model, params)

        while True:
            try:
                markers_data = self.kin_queue_in.get_nowait()
                is_working = True
            except:
                is_working = False

            if is_working:
                markers = markers_data["markers"]
                states = markers_data["states"]
                if self.try_w_connection:
                    markers_tmp = markers_data["markers_tmp"]
                else:
                    markers_tmp = self.markers_exp[:, :, self.m : self.m + 1]
                    self.m = self.m + 1 if self.m < self.last_frame else self.init_frame

                markers_tmp = markers_tmp * 0.001
                if self.recons_kalman:
                    states_tmp = self.kalman_func(markers_tmp, model, return_q_dot=False, kalman=kalman)

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

    def save_streamed_data(self):
        emg_dec = self.emg_dec
        markers_dec = self.markers_dec
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
            if self.device == "vicon":
                self._init_vicon_client()
        self.nb_marks = len(self.marker_names)
        if self.plot_emg:
            p, win_emg, app, box = init_plot_emg(self.nb_electrodes)
        delta = 0
        delta_tmp = 0
        self.iter = 0
        dic_to_put = {}
        frame = False

        while True:
            tic = time()
            if self.iter == 0:
                initial_time = time() - tic
                print("Data start streaming")

            if self.try_w_connection:
                if self.device == "vicon":
                    frame = self.vicon_client.GetFrame()
                    absolute_time_frame = datetime.datetime.now()  # time at wich data are received
                    vicon_latency_total = self.vicon_client.GetLatencyTotal()
                    if frame is not True:
                        print("A problem occurred, no frame available.")

                elif self.device == "pytrigno":
                    frame = True
                    absolute_time_frame = datetime.datetime.now()
            else:
                frame = True
                absolute_time_frame = datetime.datetime.now()

            absolute_time_frame_dic = {"day": absolute_time_frame.day,
                                       "hour": absolute_time_frame.hour,
                                       "hour_s": absolute_time_frame.hour * 3600,
                                       "minute": absolute_time_frame.minute,
                                       "minute_s": absolute_time_frame.minute * 60,
                                       "second": absolute_time_frame.second,
                                       "millisecond": int(absolute_time_frame.microsecond/1000),
                                       "millisecond_s": int(absolute_time_frame.microsecond / 1000) * 0.001,
                                       }
            if frame:
                if self.stream_emg:
                    if self.try_w_connection:
                        emg_tmp, emg_names = self.get_emg(emg_names=self.emg_names)
                    else:
                        emg_tmp = []
                        emg_names = []
                    self.emg_queue_in.put_nowait({"raw_emg": raw_emg, "emg_proc": emg_proc, "emg_tmp": emg_tmp})
                if self.stream_markers:
                    if self.try_w_connection:
                        markers_tmp, self.marker_names, occluded = self.get_markers()
                    else:
                        markers_tmp = []
                    self.kin_queue_in.put_nowait(
                        {
                            "states": states,
                            "markers": markers,
                            "model_path": self.model_path,
                            "markers_tmp": markers_tmp,
                        }
                    )
                if self.stream_imu:
                    if self.try_w_connection:
                        imu_tmp, imu_names = self.get_imu(imu_names=self.imu_names)
                    else:
                        imu_tmp, imu_names = [], []
                    self.imu_queue_in.put_nowait({"raw_imu": raw_imu, "imu_proc": imu_proc, "imu_tmp": imu_tmp})

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
                    dic_to_put["imu_names"] = imu_names
                    dic_to_put["raw_imu"] = raw_imu
                    dic_to_put["imu_proc"] = imu_proc

            dic_to_put["acquisition_rate"] = self.acquisition_rate
            dic_to_put["absolute_time_frame"] = absolute_time_frame_dic
            if self.device == "vicon":
                dic_to_put["vicon_latency"] = vicon_latency_total
            process_time = time() - tic  # time to process all data + time to get data
            for i in range(len(self.ports) + len(self.osc_ports)):
                try:
                    self.server_queue[i].get_nowait()
                except:
                    pass
                self.server_queue[i].put_nowait(dic_to_put)

            if self.plot_emg:
                update_plot_emg(raw_emg, p, app, box)

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
                    data_to_save["raw_emg"] = raw_emg[:, -self.emg_sample :]

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

                add_data_to_pickle(data_to_save, self.data_path)

            duration = time() - tic
            if 1 / duration > self.acquisition_rate:
                sleep((1 / self.acquisition_rate) - duration)
            # delta, delta_tmp = self._loop_sleep(delta_tmp, delta, tic)

    def get_markers(self, markers_names=()):
        occluded = []
        markers = self.markers_empty
        subject_name = self.subject_name
        marker_names = markers_names if len(markers_names) != 0 else self.vicon_client.GetMarkerNames(subject_name)
        for i in range(len(marker_names)):
            if markers_names:
                markers[:, i, 0], occluded_tmp = self.vicon_client.GetMarkerGlobalTranslation(
                    subject_name, marker_names[i]
                )
            else:
                markers[:, i, 0], occluded_tmp = self.vicon_client.GetMarkerGlobalTranslation(
                    subject_name, marker_names[i][0]
                )
                marker_names[i] = marker_names[i][0]
            occluded.append(occluded_tmp)
        return markers, marker_names, occluded

    @staticmethod
    def get_force_plate(vicon_client):
        forceVectorData = []
        forceplates = vicon_client.GetForcePlates()
        for plate in forceplates:
            forceVectorData = vicon_client.GetForceVector(plate)
            momentVectorData = vicon_client.GetMomentVector(plate)
            copData = vicon_client.GetCentreOfPressure(plate)
            globalForceVectorData = vicon_client.GetGlobalForceVector(plate)
            globalMomentVectorData = vicon_client.GetGlobalMomentVector(plate)
            globalCopData = vicon_client.GetGlobalCentreOfPressure(plate)

            try:
                analogData = vicon_client.GetAnalogChannelVoltage(plate)
            except VDS.DataStreamException as e:
                print("Failed getting analog channel voltages")
        return forceVectorData

    def get_emg(self, emg_names=None):  # init=False, output_names=None, emg_names=None):
        # output_names = [] if output_names is None else output_names
        names = [] if emg_names is None else emg_names
        if self.device == "vicon":
            emg = np.zeros((16, self.emg_sample))
            # if init is True:
            #     count = 0
            #     for output_name, emg_name, unit in self.device_info:
            #         emg[count, :], occluded = self.vicon_client.GetDeviceOutputValues(
            #             self.device_name, output_name, emg_name
            #         )
            #         if np.mean(emg[count, -self.emg_sample :]) != 0:
            #             output_names.append(output_name)
            #             emg_names.append(emg_name)
            #         count += 1
            # else:
            #     for i in range(len(output_names)):
            #         emg[i, :], occluded = self.vicon_client.GetDeviceOutputValues(
            #             self.device_name, output_names[i], emg_names[i]
            #         )
            count = 0
            for output_name, emg_name, unit in self.device_info:
                emg[count, :], occluded = self.vicon_client.GetDeviceOutputValues(
                    self.device_name, output_name, emg_name
                )
                if emg_names is None:
                    names.append(emg_name)
                count += 1
            emg = emg[: self.nb_electrodes, :]
        else:
            emg = self.dev_emg.read()

        # if init is True:
        #     return output_names, emg_names
        # else:
        return emg, names

    def get_imu(self, imu_names=None):  # , init=False, output_names=None, imu_names=None):
        # output_names = [] if output_names is None else output_names
        names = [] if imu_names is None else imu_names
        if self.device == "vicon":
            imu = np.zeros((144, self.imu_sample))
            # if init is True:
            #     count = 0
            #     for output_name, imu_name, unit in self.imu_device_info:
            #         imu_tmp, occluded = self.vicon_client.GetDeviceOutputValues(
            #             self.imu_device_name, output_name, imu_name
            #         )
            #         imu[count, :] = imu_tmp[-self.imu_sample:]
            #         if np.mean(imu[count, :, -self.imu_sample:]) != 0:
            #             output_names.append(output_name)
            #             imu_names.append(imu_name)
            #         count += 1
            # else:
            count = 0
            for output_name, imu_name, unit in self.imu_device_info:
                imu_tmp, occluded = self.vicon_client.GetDeviceOutputValues(self.imu_device_name, output_name, imu_name)
                if imu_names is None:
                    names.append(imu_name)
                imu[count, :] = imu_tmp[-self.imu_sample :]
                count += 1

            imu = imu[: self.nb_electrodes * 9, :]
            imu = imu.reshape(self.nb_electrodes, 9, -1)
        else:
            imu = self.dev_imu.read()
            imu = imu.reshape(self.nb_electrodes, 9, -1)

        return imu, names

    @staticmethod
    def kalman_func(markers, model, return_q_dot=True, kalman=None):
        markers_over_frames = []
        if not kalman:
            freq = 100  # Hz
            params = biorbd.KalmanParam(freq)
            kalman = biorbd.KalmanReconsMarkers(model, params)

        q = biorbd.GeneralizedCoordinates(model)
        q_dot = biorbd.GeneralizedVelocity(model)
        qd_dot = biorbd.GeneralizedAcceleration(model)
        for i in range(markers.shape[2]):
            markers_over_frames.append([biorbd.NodeSegment(m) for m in markers[:, :, i].T])

        q_recons = np.ndarray((model.nbQ(), len(markers_over_frames)))
        q_dot_recons = np.ndarray((model.nbQ(), len(markers_over_frames)))
        for i, targetMarkers in enumerate(markers_over_frames):
            kalman.reconstructFrame(model, targetMarkers, q, q_dot, qd_dot)
            q_recons[:, i] = q.to_array()
            q_dot_recons[:, i] = q_dot.to_array()

        # compute markers from
        if return_q_dot:
            return q_recons, q_dot_recons
        else:
            return q_recons
