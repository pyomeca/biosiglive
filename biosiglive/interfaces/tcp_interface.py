"""
This file is part of biosiglive. It contains a wrapper to use a client more easily.
"""

import numpy as np
from .param import Device, MarkerSet
from ..streaming.client import Client, Message
from typing import Union
try:
    from vicon_dssdk import ViconDataStream as VDS
except ModuleNotFoundError:
    pass


class TcpClient():
    """
    Class for interfacing with the client.
    """
    def __init__(self, ip: str = None, port: int = 801, type: str = "TCP", read_frequency: int = 100):
        """
        Initialize the client.
        Parameters
        ----------
        ip: str
            IP address of the server.
        port: int
            Port of the server.
        type: str
            Type of the server.
        read_frequency: int
            Frequency of the reading of the data.
        """

        ip = ip if ip else "127.0.0.1"
        self.devices = []
        self.imu = []
        self.markers = []
        self.data_to_stream = []
        self.read_frequency = read_frequency
        self.ip = ip
        self.port = port
        self.message = Message(read_frequency=read_frequency)
        self.client = Client(server_ip=ip, port=port, type=type)

    def add_device(self, name: str, type: str = "emg", rate: float = 2000, system_rate: float = 100):
        """
        Add a device to the client.
        Parameters
        ----------
        name: str
            Name of the device.
        type: str
            Type of the device. (emg, imu, etc.)
        rate: float
            Frequency of the device.
        system_rate: float
            Acquisition frequency.
        """
        device_tmp = Device(name, type, rate, system_rate)
        self.devices.append(device_tmp)
        # self.message.add_command(name="command", value=type)
        self.message.command.append(type)

    def set_message(self, message: Message):
        self.message = message

    def get_message(self):
        return self.message

    def add_markers(self, name: str = None, rate: int = 100, unlabeled: bool = False, subject_name: str = None):
        """
        Add a marker set to the client.
        Parameters
        ----------
        name: str
            Name of the marker set.
        rate: int
            Frequency of the marker set.
        unlabeled: bool
            If the marker set is unlabeled.
        subject_name: str
            Name of the subject.
        """
        markers_tmp = MarkerSet(name, rate, unlabeled)
        markers_tmp.subject_name = subject_name
        markers_tmp.markers_names = name
        self.markers.append(markers_tmp)
        # self.message.add_command(name="command", value="markers")
        self.message.command.append("markers")

    def get_data_from_server(self):
        """
        Get the data from the server.
        Returns
        -------
        data: list
            ALL the data asked from the server.
        """
        all_data = []
        data = self.client.get_data(message=self.message)
        for stream_data in self.data_to_stream:
            for key in data:
                if key == stream_data:
                    all_data.append(np.array(data[key]))
        return all_data

    def get_device_data(self, device_name: str = "all", get_names: bool = False, *args):
        """
        Get the data from a device.
        Parameters
        ----------
        device_name: str
            Name of the device. all for all the devices.
       get_names: bool
            If the names of the devices should be returned.

        Returns
        -------
        data: list
            The data asked from the server.
        """
        devices = []
        all_device_data = []
        if not isinstance(device_name, list):
            device_name = [device_name]

        if device_name != "all":
            for d, device in enumerate(self.devices):
                if device.name == device_name[d]:
                    devices.append(device)
        else:
            devices = self.devices

        self.message.update_command(name="get_names", value=get_names)
        self.message.update_command(name="command", value=[i.type for i in devices])
        data = self.client.get_data(self.message)
        for device in devices:
            for key in data:
                if key == device.type:
                    all_device_data.append(np.array(data[key]))
        return all_device_data

    def get_markers_data(self, get_names: bool = False):
        """
        Get the data from the markers.

        Parameters
        ----------
        get_names: bool
            If the names of the markers should be returned.

        Returns
        -------
        data: list
            The data asked from the server.
        """
        self.message.update_command(name="get_names", value=get_names)
        self.message.update_command(name="command", value="markers")
        data = self.client.get_data(self.message)
        return np.array(data["markers"])




