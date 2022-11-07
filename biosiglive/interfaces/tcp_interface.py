"""
This file is part of biosiglive. It contains a wrapper to use a tcp client more easily.
"""

import numpy as np
from ..streaming.client import Client, Message
from .generic_interface import GenericInterface
from ..enums import DeviceType, InterfaceType
from typing import Union

try:
    from vicon_dssdk import ViconDataStream as VDS
except ModuleNotFoundError:
    pass


class TcpClient(GenericInterface):
    """
    Class for interfacing with the client.
    """

    def __init__(self, ip: str = "127.0.0.1", port: int = 801, client_type: str = "TCP", read_frequency: int = 100):
        """
        Initialize the client.
        Parameters
        ----------
        ip: str
            IP address of the server.
        port: int
            Port of the server.
        client_type: str
            Type of the server.
        read_frequency: int
            Frequency of the reading of the data.
        """
        super(TcpClient, self).__init__(ip, interface_type=InterfaceType.TcpClient)
        self.devices = []
        self.imu = []
        self.markers = []
        self.data_to_stream = []
        self.read_frequency = read_frequency
        self.ip = ip
        self.port = port
        self.message = Message(read_frequency=read_frequency)
        self.client = Client(server_ip=ip, port=port, client_type=client_type)

    def add_device(
        self,
        name: str,
        device_type: Union[DeviceType, str] = DeviceType.Emg,
        rate: float = 2000,
        device_range: tuple = (0, 16),
    ):
        """
        Add a device to the client.
        Parameters
        ----------
        name: str
            Name of the device.
        device_type: Union[DeviceType, str]
            Type of the device. (emg, imu, etc.)
        rate: float
            Frequency of the device.
        device_range: tuple
            Range of the device.
        """
        device_tmp = self._add_device(name, device_type, rate, device_range)
        device_tmp.interface = self.interface_type
        self.devices.append(device_tmp)
        self.message.command.append(device_type.value)

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
        if len(self.markers) != 0:
            raise ValueError("Only one marker set can be added for now.")
        markers_tmp = self._add_markers(name, rate, unlabeled)
        markers_tmp.subject_name = subject_name
        markers_tmp.interface = self.interface_type
        self.markers.append(markers_tmp)
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

    def get_device_data(self, device_name: str = "all", get_names: bool = False):
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
                if key == device.device_type:
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
