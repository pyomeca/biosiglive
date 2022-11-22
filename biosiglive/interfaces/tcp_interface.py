"""
This file is part of biosiglive. It contains a wrapper to use a tcp client more easily.
"""

import numpy as np
from ..streaming.client import Client, Message
from .generic_interface import GenericInterface
from ..enums import DeviceType, InterfaceType, RealTimeProcessingMethod, OfflineProcessingMethod, InverseKinematicsMethods
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
        self.marker_sets = []
        self.data_to_stream = []
        self.read_frequency = read_frequency
        self.ip = ip
        self.port = port
        self.client = Client(server_ip=ip, port=port, client_type=client_type)

    def add_device(
        self,
        nb_channels: int,
        device_type: Union[DeviceType, str] = DeviceType.Emg,
        name: str = None,
        rate: float = 2000,
        device_range: tuple = None,
        process_method: Union[RealTimeProcessingMethod, OfflineProcessingMethod] = None,
        **process_kwargs
    ):
        """
        Add a device to the client.
        Parameters
        ----------
        nb_channels: int
            Number of channels of the device.
        device_type: Union[DeviceType, str]
            Type of the device. (emg, imu, etc.)
        name: str
            Name of the device.
        rate: float
            Frequency of the device.
        device_range: tuple
            Range of the device.
        process_method: Union[RealTimeProcessingMethod, OfflineProcessingMethod]
            Method to process the data.
        process_kwargs: dict
            Keyword arguments for the processing method.
        """
        device_tmp = self._add_device(nb_channels, name, device_type, rate, device_range, process_method, **process_kwargs)
        device_tmp.interface = self.interface_type
        self.devices.append(device_tmp)

    def add_marker_set(
        self,
        nb_markers: int,
        name: str = None,
        marker_names: Union[str, list] = None,
        rate: float = 100,
        unlabeled: bool = False,
        subject_name: str = None,
        kinematics_method: InverseKinematicsMethods = None,
        **kin_method_kwargs,
    ):
        """
        Add markers set to stream from the Vicon system.
        Parameters
        ----------
        nb_markers: int
            Number of markers.
        name: str
            Name of the markers set.
        marker_names: Union[list, str]
            List of markers names.
        rate: int
            Rate of the markers set.
        unlabeled: bool
            Whether the markers set is unlabeled.
        subject_name: str
            Name of the subject. If None, the subject will be the first one in Nexus.
        kinematics_method: InverseKinematicsMethods
            Method used to compute the kinematics.
        **kin_method_kwargs
            Keyword arguments for the kinematics method.
        """
        if len(self.marker_sets) != 0:
            raise ValueError("Only one marker set can be added for now.")

        markers_tmp = self._add_marker_set(
            nb_markers=nb_markers,
            name=name,
            marker_names=marker_names,
            rate=rate,
            unlabeled=unlabeled,
            kinematics_method=kinematics_method,
            **kin_method_kwargs,
        )
        self.marker_sets.append(markers_tmp)

    def get_data_from_server(self, command: str = "all", nb_frame_to_get: int = None, down_sampling: dict = None):
        """
        Get the data from the server.
        Returns
        -------
        data: list
            ALL the data asked from the server.
        """
        all_data = []
        data = self.client.get_data(
            message=Message(command=command, nb_frame_to_get=nb_frame_to_get, down_sampling=down_sampling)
        )
        for stream_data in self.data_to_stream:
            for key in data:
                if key == stream_data:
                    all_data.append(np.array(data[key]))
        return all_data

    def get_device_data(self, device_name: str = "all"):
        """
        Get the data from a device.

        Parameters
        ----------
        device_name: str
            Name of the device. all for all the devices.

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
        data = self.client.get_data(self.message)
        for device in devices:
            for key in data:
                if key == device.device_type:
                    all_device_data.append(np.array(data[key]))
        return all_device_data

    def get_marker_set_data(self, get_names: bool = False):
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
