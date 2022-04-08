"""
This file is part of biosiglive. It allows connecting to a biosiglive server and to receive data from it.
"""

import socket
import json
import struct

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
            "nb_of_data_to_export": 1
        }


class Client:
    def __init__(self, server_address: str, port: int, type: str = "TCP", name: str = None):
        """
        Create a client socket.
        Parameters
        ----------
        server_address: str
            Server address.
        port: int
            Server port.
        type: str
            Type of the socket.
        name: str
            Name of the client.
        """

        self.name = name if name is not None else "Client"
        self.type = type
        self.address = f"{server_address}:{port}"
        if self.type == "TCP":
            self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        elif self.type == "UDP":
            self.client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        else:
            raise RuntimeError(f"Invalid type of connexion ({ self.type}). Type must be 'TCP' or 'UDP'.")
        self.client.connect((server_address, port))

    @staticmethod
    def client_sock(type: str,):
        """
        Create a client socket.
        Parameters
        ----------
        type: str
            Type of the socket.

        Returns
        -------
        client: socket
            Client socket.
        """
        if type == "TCP" or type is None:
            return socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        elif type == "UDP":
            return socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

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

    def get_data(
        self,
        data: list,
        read_frequency: int = 100,
        emg_wind: int = 2000,
        nb_of_data_to_export: int = 1,
        buff: int = Buff_size,
        get_kalman=False,
        get_names=False,
        mvc_list: list = None,
        norm_emg: bool = False,
        raw: bool = False,
    ):
        """
        Get data from the server.
        Parameters
        ----------
        data: list
            List of data to send.
        read_frequency: int
            Frequency of the data.
        emg_wind: int
            Size of the EMG window.
        nb_of_data_to_export: int
            Number of data to export.
        buff: int
            Size of the buffer.
        get_kalman: bool
            Get Kalman data.
        get_names: bool
            Get names of the channels.
        mvc_list: list
            List of MVC.
        norm_emg: bool
            Normalize EMG.
        raw: bool
            Get raw data.

        Returns
        -------
        data: dict
            Data received.
        """

        message = Message()
        message.dic["get_names"] = get_names
        message.dic["norm_emg"] = norm_emg
        message.dic["mvc_list"] = mvc_list
        message.dic["kalman"] = get_kalman
        message.dic["read_frequency"] = read_frequency
        message.dic["emg_windows"] = emg_wind
        message.dic["nb_of_data_to_export"] = nb_of_data_to_export
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

        self.client.sendall(json.dumps(message.dic).encode())
        return self.recv_all(buff)
