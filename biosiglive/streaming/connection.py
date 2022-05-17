"""
This is part of the biosiglive project. It contains the connection class.
"""
import socket
import json
import multiprocessing as mp
import numpy as np
import struct
from typing import Union
try:
    from pythonosc.udp_client import SimpleUDPClient
except ModuleNotFoundError:
    pass


class Connection:
    """
    This class is used to connect to the biosiglive server.
    """
    def __init__(self, ip: str = "127.0.0.1", ports: Union[int, list] = 50000):
        """
        Initialize the connection.

        Parameters
        ----------
        ip : str
            The ip address of the server.
        ports : int or list
            The port(s) of the server.
        """
        self.ip = ip
        self.ports = [ports] if not isinstance(ports, list) else ports
        self.message_queues = None
        self.buff_size = 100000
        self.acquisition_rate = 100

    def _prepare_data(self, message: dict, data: dict):
        """
        Prepare the data to send.

        Parameters
        ----------
        message : dict
            The message received from the client.
        data : dict
            The data to prepared.

        Returns
        -------
        prepared data : dict
            The data prepared to be sent.

        """
        read_frequency = message["read_frequency"]
        raw_data = message["raw_data"]
        nb_frames_to_get = message["nb_frames_to_get"] if message["nb_frames_to_get"] else 1

        if self.acquisition_rate < read_frequency:
            raise RuntimeError(f"Acquisition rate ({self.acquisition_rate}) is lower than read "
                               f"frequency ({read_frequency}).")
        else:
            ratio = int(self.acquisition_rate / read_frequency)
        data_to_prepare = self.__data_to_prepare(message, data)
        prepared_data = self.__check_and_adjust_dims(data_to_prepare, ratio, raw_data, nb_frames_to_get)
        if message["get_names"]:
            prepared_data["marker_names"] = data["marker_names"]
            prepared_data["emg_names"] = data["emg_names"]
        if "absolute_time_frame" in data.keys():
            prepared_data["absolute_time_frame"] = data["absolute_time_frame"]
        return prepared_data

    @staticmethod
    def __data_to_prepare(message: dict, data: dict):
        """
        Prepare the device data to send.

        Parameters
        ----------
        message : dict
            The message received from the client.
        data : dict
            The data to prepared.

        Returns
        -------
        prepared data : dict
            The data prepared to be sent.
        """

        data_to_prepare = {}
        if len(message["command"]) != 0:
            for i in message["command"]:
                if i == "emg":
                    if message["raw_data"]:
                        raw_emg = data["raw_emg"]
                        data_to_prepare["raw_emg"] = raw_emg
                    emg = data["emg_proc"]
                    if message["norm_emg"]:
                        if isinstance(message["mvc_list"], np.ndarray) is True:
                            if len(message["mvc_list"].shape) == 1:
                                quot = message["mvc_list"].reshape(-1, 1)
                            else:
                                quot = message["mvc_list"]
                        else:
                            quot = np.array(message["mvc_list"]).reshape(-1, 1)
                    else:
                        quot = [1]
                    data_to_prepare["emg"] = emg / quot

                elif i == "markers":
                    markers = data["markers"]
                    data_to_prepare["markers"] = markers

                elif i == "imu":
                    if message["raw_data"]:
                        raw_imu = data["raw_imu"]
                        data_to_prepare["raw_imu"] = raw_imu
                    imu = data["imu_proc"]
                    data_to_prepare["imu"] = imu

                elif i == "force plate":
                    raise RuntimeError("force plate not implemented yet.")
                else:
                    raise RuntimeError(
                        f"Unknown command '{i}'. Command must be :'emg', 'markers' or 'imu' "
                    )
        else:
            raise RuntimeError(f"No command received.")

        return data_to_prepare

    @staticmethod
    def __check_and_adjust_dims(data: dict, ratio: int, raw_data: bool = False, nb_frames_to_get: int = 1):
        """
        Check and adjust the dimensions of the data to send.

        Parameters
        ----------
        data : dict
            The data to check and adjust.
        ratio : int
            The ratio between the acquisition rate and the read frequency.
        raw_data : bool
            If the raw data must be sent (default is False).
        nb_frames_to_get : int
            The number of frames to get (default is 1).

        Returns
        -------
        data : dict
            The data checked and adjusted.
        """

        for key in data.keys():
            if "sample" not in key:
                if len(data[key].shape) == 2:
                    if key != "raw_emg":
                        data[key] = data[key][:, ::ratio]
                    if raw_data and key == "raw_emg":
                        nb_frames_to_get = data["emg_sample"] * nb_frames_to_get
                    data[key] = data[key][:, -nb_frames_to_get:].tolist()
                elif len(data[key].shape) == 3:
                    if key != "raw_imu":
                        data[key] = data[key][:, :, ::ratio]
                    if raw_data and key == "raw_imu":
                        nb_frames_to_get = data["imu_sample"] * nb_frames_to_get
                    data[key] = data[key][:, :, -nb_frames_to_get:].tolist()
        return data


class Server(Connection):
    """
    Class to create a server.
    """
    def __init__(self, ip: str = "127.0.0.1", port: int = 50000, type: str = "TCP"):
        """
        Parameters
        ----------
        ip : str
            The ip of the server.
        ports : list
                The ports of the server.
        type : str
            The type of the server.
        """
        self.ip = ip
        self.port = port
        self.type = type
        self.server = None
        self.inputs = None
        self.outputs = None
        self.buff_size = 100000
        super().__init__(ip=ip, ports=port)

    def start(self):
        """
        Start the server.
        """
        # for i, port in enumerate(self.ports):
        if self.type == "TCP":
            self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        elif self.type == "UDP":
            raise RuntimeError(f"UDP server not implemented yet.")
            # self.servers.append(socket.socket(socket.AF_INET, socket.SOCK_DGRAM))
        else:
            raise RuntimeError(f"Invalid type of connexion ({type}). Type must be 'TCP' or 'UDP'.")
        try:
            self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server.bind((self.ip, self.port))
            if self.type != "UDP":
                self.server.listen(10)
                self.inputs = [self.server]
                self.outputs = []
                self.message_queues = {}
        except ConnectionError:
            raise RuntimeError("Unknown error. Server is not listening.")

    def client_listening(self, data: dict):
        """
        Listen to the client.

        Parameters
        ----------
        data : dict
            Data to send to the client function of message
        """
        connection, ad = self.server.accept()
        message = json.loads(connection.recv(self.buff_size))
        data_to_send = self._prepare_data(message, data)
        self._send_data(data_to_send, connection)

    @staticmethod
    def _send_data(data, connection):
        """
        Send the data to the client.

        Parameters
        ----------
        data : dict
            The data to send.
        connection : socket.socket
            The connection to send the data to.
        """
        # if self.optim is not True:
        #     print("Sending data to client...")
        encoded_data = json.dumps(data).encode()
        encoded_data = struct.pack('>I', len(encoded_data)) + encoded_data
        try:
            connection.sendall(encoded_data)
            print(f"data sended : {data}")
        except ConnectionError:
            pass


class OscClient(Connection):
    """
    Class to create an OSC client.
    """
    def __init__(self, ip: str = "127.0.0.1"):
        self.ports = [51337]
        self.osc = []
        super().__init__(ip=ip, ports=self.ports)

    def start(self):
        """
        Start the client.
        """
        for i in range(len(self.ports)):
            try:
                self.osc.append(SimpleUDPClient(self.ip, self.ports[i]))
                print(f"Streaming OSC {i} activated on '{self.ip}:{self.ports[i]}")
            except ConnectionError:
                raise RuntimeError("Unknown error. OSC client not open.")

    @staticmethod
    def __adjust_dims(data: dict, device_to_send: dict):
        """
        Adjust the dimensions of the data to send.

        Parameters
        ----------
        data : dict
            The data to send.
        device_to_send : dict
            The device type to send the data to (emg or imu).
        Returns
        -------
        The data to send.
        """
        data_to_return = []
        for key in device_to_send:
            if key == "emg":
                emg_proc = np.array(data["emg_proc"])[:, -1:]
                emg_proc = emg_proc.reshape(emg_proc.shape[0])
                data_to_return.append(emg_proc.tolist())
            elif key == "imu":
                imu = np.array(data["imu_proc"])[:, :, -1:]
                data_to_return.append(imu.tolist())
                accel_proc = imu[:, :3, :]
                accel_proc = accel_proc.reshape(accel_proc.shape[0])
                data_to_return.append(accel_proc.tolist())
                gyro_proc = imu[:, 3:6, :]
                gyro_proc = gyro_proc.reshape(gyro_proc.shape[0])
                data_to_return.append(gyro_proc.tolist())
            else:
                raise RuntimeError(f"Unknown device ({key}) to send. Possible devices are 'emg' and 'imu'.")

        return data_to_return

    def send_data(self, data: dict, device_to_send: dict):
        """
        Send the data to the client.
        Parameters
        ----------
        data : dict
            The data to send.
        device_to_send : dict
            The device type to send the data to (emg or imu).
        """

        data = self.__adjust_dims(data, device_to_send)
        for key in device_to_send:
            if key == "emg":
                self.osc[0].send_message("/emg", data[0])
            elif key == "imu":
                idx = 1 if key in "emg" in device_to_send else 0
                self.osc[0].send_message("/imu/", data[idx])
                self.osc[0].send_message("/accel/", data[idx + 1])
                self.osc[0].send_message("/gyro/", data[idx + 2])
            else:
                raise RuntimeError(f"Unknown device ({key}) to send. Possible devices are 'emg' and 'imu'.")

