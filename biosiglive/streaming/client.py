"""
This file is part of biosiglive. It allows connecting to a biosiglive server and to receive data from it.
"""

import socket
import json
import struct
from typing import Union

Buff_size = 32767


class Message:
    def __init__(self,
                 command: list = (),
                 read_frequency: float = 100,
                 nb_frame_to_get: int = 1,
                 get_names: bool = None,
                 mvc_list: list = None,
                 kalman: bool = None,
                 get_raw_data: bool = False):
        """
        Message class
        """

        self.command = []
        self.command = command
        self.read_frequency = 100
        self.emg_windows = 2000
        self.get_names = False
        self.nb_frames_to_get = 1
        self.get_names = get_names
        self.mvc_list = mvc_list
        self.kalman = kalman
        self.read_frequency = read_frequency
        self.nb_frames_to_get = nb_frame_to_get
        self.raw_data = get_raw_data

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
            self.__setattr__(name, values[i])

    def get_command(self):
        """
        Get the command.

        Returns
        -------
        message: Message.dic
            Message containing the command.
        """
        return self.command

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
        self.message = Message()
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

    def get_data(self, message: Message = Message(), buff: int = Buff_size):
        """
        Get the data from server using the command.

        Parameters
        ----------
        message
        buff: int
            Size of the buffer.

        Returns
        -------
        data: dict
            Data from server.
        """

        self._connect()
        self.client.sendall(json.dumps(message.__dict__).encode())
        return self.recv_all(buff)

