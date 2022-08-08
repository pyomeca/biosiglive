
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
from biosiglive.io.save_data import read_data

vicon_package, biorbd_package = True, True

try:
    import biorbd
except ModuleNotFoundError:
    biorbd_package = False

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
            stream_rate=None,
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
        self.stream_rate = stream_rate if stream_rate else acquisition_rate
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
        self.markers_dec = 5
        self.emg_dec = 8
        self.buff_size = buff_size
        self.save_data = save_data
        self.raw_data = False
        self.try_w_connection = True
        self.device_host_ip = device_host_ip
        self.offline_file_path = None
        self.smooth_markers = None

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

        self.nb_electrodes = (electrode_idx[1] - electrode_idx[0]) + 1 if isinstance(electrode_idx, tuple) else 1
        if norm and not mvc:
            raise RuntimeError("Please provide mvc data to normalize emg signals or turn 'norm' to False")
        self.plot_emg = live_plot
        self.mvc_list = mvc
        self.norm_emg = norm
        self.stream_emg = True
        self.interface.add_device(name=device_name, range=electrode_idx, type="emg", rate=rate)
        self.interface.devices[-1].process_method = None if not process else self.interface.devices[-1].process_method
        self.emg_sample = self.interface.devices[-1].sample
        self.emg_rate = self.interface.devices[-1].rate

    def add_markers(self, nb_markers: int,
                    marker_set_name: str = None,
                    subject: str = None,
                    smooth_traj: bool = None,
                    rate: float = 100,
                    unlabeled: bool = False,
                    compute_kin: bool = False,
                    msk_model: str = None,
                    window = None):
        self.stream_markers = True
        self.nb_marks = nb_markers
        self.smooth_markers = smooth_traj
        self.recons_kalman = compute_kin
        self.model_path = msk_model
        self.markers_windows = window if window else rate
        if self.stream_from == "pytrigno":
            raise RuntimeError("Impossible to stream markers data from pytrigno.")
        else:
            self.interface.add_markers(name=marker_set_name, rate=rate, unlabeled=unlabeled, subject_name=subject)

        self.markers_rate = self.interface.markers[-1].rate
        self.markers_sample = self.interface.markers[-1].sample

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
            try:
                data_exp = sio.loadmat(self.offline_file_path)
            except:
                data_exp = read_data(self.offline_file_path)

        if self.try_w_connection is not True:
            if self.stream_emg:
                self.emg_sample = int(self.emg_rate / self.acquisition_rate)
                if self.offline_file_path and "raw_emg" in data_exp.keys():
                    self.emg_exp = data_exp["raw_emg"]

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
        # self.open_server()
        processes = [self.process(name="reader", target=LiveData.save_streamed_data, args=(self,))]
        for i in range(len(self.server_ports)):
            processes.append(self.process(name="listen" + f"_{i}", target=LiveData.open_server, args=(self,)))
        if self.stream_emg:
            processes.append(self.process(name="process_emg", target=LiveData.emg_processing, args=(self,)))
        if self.stream_markers:
            processes.append(self.process(name="kin", target=LiveData.recons_kin, args=(self,)))
        for p in processes:
            p.start()
            if p.name.startswith("listen"):
                self.count_server += 1
        for p in processes:
            p.join()

    def open_server(self):
        server = Server(self.server_ip, self.server_ports[self.count_server], type=self.type)
        server.start()
        while True:
            connection, message = server.client_listening()
            data_queue = []
            while len(data_queue) == 0:
                try:
                    data_queue = self.server_queue[self.count_server].get_nowait()
                    is_working = True
                except:
                    is_working = False
                    pass
                if is_working:
                    server.send_data(data_queue, connection, message)

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
            except:
                is_working = False

            if is_working:
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

    def recons_kin(self):
        model, kalman, markers_data = None, None, None
        model = biorbd.Model(self.model_path)
        if self.recons_kalman:
            freq = self.markers_rate  # Hz
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
                markers_tmp = markers_data["markers_tmp"]
                if self.recons_kalman:
                    q_tmp, dq_tmp = kalman_func(markers_tmp, model, return_q_dot=True, kalman=kalman, use_kalman=True)
                    states_tmp = np.concatenate((q_tmp, dq_tmp), axis=0)
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
            absolute_time_frame_dic = {"day": absolute_time_frame.day,
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
                        emg_tmp = self.emg_exp[: self.nb_electrodes, c: c + self.emg_sample]
                        c = c + self.emg_sample
                        # c = c + self.emg_sample if c + self.emg_sample < self.last_frame else self.init_frame
                    self.emg_queue_in.put_nowait({"raw_emg": raw_emg, "emg_proc": emg_proc, "emg_tmp": emg_tmp})
                if self.stream_markers:
                    if self.try_w_connection:
                        all_markers_tmp, all_occluded = self.interface.get_markers_data()
                        markers_tmp = all_markers_tmp[0]
                    else:
                        markers_tmp = self.markers_exp[:, :, m: m + 1]
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
                        data_to_save["raw_emg"] = raw_emg[:, -self.emg_sample:]

                    if self.stream_markers:
                        data_to_save["markers"] = markers[:3, :, -1:]

                    if self.recons_kalman:
                        data_to_save["kalman"] = states[:, -1:]
                    save_data.add_data_to_pickle(data_to_save, self.output_file_path)

                self.iter += 1
            print(time() - tic)
            if not self.try_w_connection:
                sleep(1/self.acquisition_rate)


if __name__ == '__main__':
    server_ip = "127.0.0.1"
    server_ports = [50000]
    live_stream = LiveData(server_ip=server_ip,
                           server_ports=server_ports,
                           stream_from="vicon",
                           device_host_ip="127.0.0.1",
                           acquisition_rate=200,
                           stream_rate=100,)
    mvc_list = [0.00053701,
                0.00046841,
                0.00038598,
                0.00078507,
                0.00116109,
                0.00091976,
                0.0010177,
                0.00099549,
                0.00035016,
                0.00035016,
                ]
    # mvc = sio.loadmat(f"data_final_new/subject_3/MVC_subject_3.mat")
    # #["MVC_list_max"][0]
    # mvc_list = [
    #     mvc[0],  # MVC Pectoralis sternalis
    #     mvc[1],  # MVC Deltoid anterior
    #     mvc[2],  # MVC Deltoid medial
    #     mvc[3],  # MVC Deltoid posterior
    #     mvc[4],  # MVC Biceps brachii
    #     mvc[5],  # MVC Triceps brachii
    #     mvc[6],  # MVC Trapezius superior
    #     mvc[7],  # MVC Trapezius medial
    #     mvc[8],  # MVC Trapezius inferior
    #     mvc[9],  # MVC Latissimus dorsi
    # ]
    live_stream.add_emg_device(electrode_idx=(0, 9),
                               device_name="EMG",
                               process=True,
                               norm=True,
                               mvc=mvc_list,
                               rate=2000)
    emg_processing = RealTimeProcessing()
    emg_processing.ma_win = 200
    live_stream.interface.devices[-1].set_process_method(emg_processing.process_emg)

    live_stream.add_markers(nb_markers=16,
                            subject="subject_3",
                            smooth_traj=True,
                            rate=200,
                            unlabeled=False,
                            compute_kin=False,
                            msk_model="data_final_new/subject_3/wu_scaled.bioMod"
                            )

    live_stream.run(test_with_connection=False,
                    show_log=True,
                    output_file_path="data_final_new/subject_3/data_abd_sans_poid_test_hh",
                    offline_file_path="data_final_new/subject_3/data_abd_sans_poid")
