"""
This file is part of biosiglive. It allows connecting to a biosiglive server and to receive data from it.
"""

import socket
import json
import struct
from typing import Union

Buff_size = 32767


class Message:
    def __init__(self):
        """
        Message class
        """

        self.command = []
        self.dic = {
            "command": [],
            "read_frequency": 100,
            "emg_windows": 2000,
            "get_names": False,
            "nb_frames_to_get": 1
        }


class Client:
    def __init__(self, server_ip: str, port: int, type: str = "TCP", name: str = None):
        """
        Create a client main.
        Parameters
        ----------
        server_ip: str
            Server address.
        port: int
            Server port.
        type: str
            Type of the main.
        name: str
            Name of the client.
        """

        self.name = name if name is not None else "Client"
        self.type = type
        self.address = f"{server_ip}:{port}"
        self.server_address = server_ip
        self.port = port
        self.message = None
        self.client = self.client_sock(self.type)

    def _connect(self):
        self.client.connect((self.server_address, self.port))

    @staticmethod
    def client_sock(type: str,):
        """
        Create a client main.
        Parameters
        ----------
        type: str
            Type of the main.

        Returns
        -------
        client: socket.socket
            Client main.
        """
        if type == "TCP" or type is None:
            return socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        elif type == "UDP":
            return socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        else:
            raise RuntimeError(f"Invalid type of connexion ({type}). Type must be 'TCP' or 'UDP'.")

    def recv_all(self, buff_size: int = Buff_size):
        """
        Receive all data from the server.
        Parameters
        ----------
        buff_size: int
            Size of the buffer.

        Returns
        -------
        data: list
            List of data received.
        """

        msg_len = self.client.recv(4)
        msg_len = struct.unpack('>I', msg_len)[0]
        data = []
        l = 0
        while l < msg_len:
            chunk = self.client.recv(buff_size)
            l += len(chunk)
            data.append(chunk)
        data = b"".join(data)
        data = json.loads(data)
        return data

    def init_command(
        self,
        data: list,
        read_frequency: int = 100,
        emg_wind: int = 2000,
        nb_frames_to_get: int = 1,
        get_kalman=False,
        get_names=False,
        mvc_list: list = None,
        norm_emg: bool = False,
        raw: bool = False,
    ):
        """
        Initialize the command.
        Parameters
        ----------
        data: list
            List of data to get.
        read_frequency: int
            Frequency at which the data are streamed.
        emg_wind: int
            Size of the EMG window.
        nb_frames_to_get: int
            Number of frames to get.
        get_kalman: bool
            Get the kalman data.
        get_names: bool
            Get the names of the data.
        mvc_list: list
            MVC values.
        norm_emg: bool
            Normalize the EMG.
        raw: bool
            Get the raw data.
        """

        message = Message()
        message.dic["get_names"] = get_names
        message.dic["norm_emg"] = norm_emg
        message.dic["mvc_list"] = mvc_list
        message.dic["kalman"] = get_kalman
        message.dic["read_frequency"] = read_frequency
        message.dic["emg_windows"] = emg_wind
        message.dic["nb_frames_to_get"] = nb_frames_to_get
        message.dic["raw_data"] = raw

        if norm_emg is True and mvc_list is None:
            raise RuntimeError("Define a list of MVC to normalize the EMG data.")

        elif mvc_list is not None and norm_emg is not True:
            print(
                "[WARNING] You have defined a list of MVC but not asked for normalization. "
                "Please turn norm_EMG to True tu normalize your data."
            )

        message.dic["command"] = []
        for i in data:
            message.dic["command"].append(i)
            if i != "emg" and i != "markers" and i != "imu" and i != "force_plate":
                raise RuntimeError(f"Unknown command '{i}'. Command must be :'emg', 'markers' or 'imu' ")

        self.message = message

    def update_command(self, name: Union[str, list], value: Union[bool, int, float, list, str]):
        """
        Update the command.

        Parameters
        ----------
        name: str
            Name of the command to update.
        value: bool, int, float, list, str
            Value of the command to update.
        """
        names = [name] if not isinstance(name, list) else value
        values = [value] if not isinstance(value, list) else value
        values = [values] if name == "command" else values

        for i, name in enumerate(names):
            self.message.dic[name] = values[i]

    def get_command(self):
        """
        Get the command.

        Returns
        -------
        message: Message.dic
            Message containing the command.
        """
        return self.message.dic

    def add_command(self, name: str, value: Union[bool, int, float, list, str]):
        """
        Add a command.

        Parameters
        ----------
        name: str
            Name of the command to add.
        value: bool, int, float, list, str
            Value of the command to add.
        """
        new_value = None
        old_value = self.get_command()[name]
        if isinstance(old_value, list):
            old_value.append(value)
            new_value = old_value
        elif isinstance(old_value, (bool, int, float, str)):
            new_value = value
        return self.update_command(name, new_value)

    def get_data(self, buff: int = Buff_size):
        """
        Get the data from server using the command.

        Parameters
        ----------
        buff: int
            Size of the buffer.

        Returns
        -------
        data: dict
            Data from server.
        """

        self._connect()
        self.client.sendall(json.dumps(self.message.dic).encode())
        return self.recv_all(buff)

