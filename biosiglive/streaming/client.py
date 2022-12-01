"""
This file is part of biosiglive. It allows connecting to a biosiglive server and to receive data from it.
"""
import socket
import struct
from typing import Union
import pickle

Buff_size = 32767


class Message:
    def __init__(
        self,
        command: Union[list, str] = "all",
        nb_frame_to_get: int = 1,
        down_sampling: dict = None,
        custom_cmd: Union[str, list] = None,
    ):
        """
        Message class

        Parameters
        ----------
        command: Union[list, str]
            List of commands to send to the server.
        nb_frame_to_get: int
            Number of frames to get from the server.
        down_sampling: dict
            Dictionary containing the down sampling number for data in command.
        """
        if isinstance(command, str):
            command = [command]
        self.command = command
        self.nb_frames_to_get = nb_frame_to_get
        self.down_sampling = down_sampling
        self.custom_cmd = custom_cmd


class Client:
    def __init__(self, server_ip: str, port: int, client_type: str = "TCP", name: str = None):
        """
        Create a client main.
        Parameters
        ----------
        server_ip: str
            Server address.
        port: int
            Server port.
        client_type: str
            Type of the main.
        name: str
            Name of the client.
        """

        self.name = name if name is not None else "Client"
        self.client_type = client_type
        self.address = f"{server_ip}:{port}"
        self.server_address = server_ip
        self.port = port
        self.client = self.client_sock(self.client_type)
        # self._connect()

    def _connect(self):
        self.client = self.client_sock(self.client_type)
        self.client.connect((self.server_address, self.port))

    @staticmethod
    def client_sock(
        tcp_type: str,
    ):
        """
        Create a client main.
        Parameters
        ----------
        tcp_type: str
            Type of the main.

        Returns
        -------
        client: socket.socket
            Client main.
        """
        if tcp_type == "TCP" or type is None:
            return socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        elif tcp_type == "UDP":
            return socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        else:
            raise RuntimeError(f"Invalid type of connexion ({tcp_type}). Type must be 'TCP' or 'UDP'.")

    def _recv_all(self, buff_size: int = Buff_size):
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
        msg_len = struct.unpack(">I", msg_len)[0]
        data = []
        l = 0
        while l < msg_len:
            chunk = self.client.recv(buff_size)
            l += len(chunk)
            data.append(chunk)
        data = b"".join(data)
        data = pickle.loads(data)
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
        message = message.__dict__
        self._connect()
        self.client.sendall(pickle.dumps(message))
        return self._recv_all(buff)
