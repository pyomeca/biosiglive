"""
This file contains a wrapper for the socket server to send data to the server.
"""

import socket
import numpy as np
import struct

try:
    from pythonosc.udp_client import SimpleUDPClient
except ModuleNotFoundError:
    pass
import pickle


class Connection:
    """
    This class is used to connect to the biosiglive server.
    """

    def __init__(self, ip: str = "127.0.0.1", port: int = 50000):
        """
        Initialize the connection.

        Parameters
        ----------
        ip : str
            The ip address of the server.
        port : int
            The port of the server.
        """
        self.ip = ip
        self.ports = port
        self.message_queues = None
        self.buff_size = 100000
        self.acquisition_rate = 100

    @staticmethod
    def _prepare_data(command: list, data: dict, down_sampling: dict = None, nb_frames_to_get: int = None):
        """
        Prepare the data to send.

        Parameters
        ----------
        command : dict
            The command received from the client.
        data : dict
            The data to prepared.
        down_sampling : dict
            The down sampling to apply to the data.
        nb_frames_to_get : int
            The number of frames to get.

        Returns
        -------
        prepared data : dict
            The data prepared to be sent.

        """
        data_tmp = {}
        if command == "all":
            command = list(data.keys())
        for key in command:
            if key in data.keys():
                data_tmp[key] = []
                for d in data[key]:
                    if down_sampling and key in down_sampling.keys():
                        if not isinstance(down_sampling[key], int):
                            raise ValueError("The down sampling must be an integer.")
                        d = d[..., :: down_sampling[key]]
                    if isinstance(d, np.ndarray):
                        if nb_frames_to_get:
                            data_tmp[key].append(d[..., -nb_frames_to_get:])
                        else:
                            data_tmp[key].append(d)
                    else:
                        data_tmp[key].append(d)
            else:
                raise ValueError(f"The asked data '{key}' is not in the data dictionary.")
            for key in data_tmp.keys():
                if len(data_tmp[key]) == 1:
                    data_tmp[key] = data_tmp[key][0]
        return data_tmp


class Server(Connection):
    """
    Class to create a server.
    """

    def __init__(self, ip: str = "127.0.0.1", port: int = 50000, server_type: str = "TCP"):
        """
        Initialize the server.

        Parameters
        ----------
        ip : str
            The ip of the server.
        port : int
            The port of the server.
        server_type : str
            The type of the server.
        """
        self.ip = ip
        self.port = port
        self.server_type = server_type
        self.server = None
        self.inputs = None
        self.outputs = None
        self.buff_size = 100000
        super().__init__(ip=ip, port=port)

    def start(self):
        """
        Start the server.
        """
        if self.server_type == "TCP":
            self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        elif self.server_type == "UDP":
            raise RuntimeError(f"UDP server not implemented yet.")
            # self.servers.append(socket.socket(socket.AF_INET, socket.SOCK_DGRAM))
        else:
            raise RuntimeError(f"Invalid type of connexion ({self.server_type}). Type must be 'TCP' or 'UDP'.")
        try:
            self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server.bind((self.ip, self.port))
            if self.server_type != "UDP":
                self.server.listen(10)
                self.inputs = [self.server]
                self.outputs = []
                self.message_queues = {}
        except ConnectionError:
            raise RuntimeError("Unknown error. Server is not listening.")

    def client_listening(self):
        """
        Waiting for the client connection.
        """
        connection, ad = self.server.accept()
        message = pickle.loads(connection.recv(self.buff_size))  # Received message
        return connection, message

    def send_data(self, data: dict, connection: socket.socket, message: dict = None):
        """
        Send the data to the client.

        Parameters
        ----------
        data : dict
            The data to send.
        connection : socket.socket
            The connection to send the data to.
        message : dict
            The message received from the client.
        """
        if message:
            if (
                "command" in message.keys()
                and "down_sampling" in message.keys()
                and "nb_frames_to_get" in message.keys()
            ):
                data = self._prepare_data(
                    message["command"], data, message["down_sampling"], message["nb_frames_to_get"]
                )
            else:
                raise ValueError(
                    "The message should be a dictionary created from the Message class or contains the key"
                    " : 'command', down_sampling, nb_frames_to_get."
                )
        encoded_data = pickle.dumps(data)
        encoded_data = struct.pack(">I", len(encoded_data)) + encoded_data
        try:
            connection.sendall(encoded_data)
        except ConnectionError:
            raise RuntimeError("Unknown error. Data not sent.")


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
