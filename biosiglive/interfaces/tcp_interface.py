"""
This file is part of biosiglive. It contains a wrapper to use a tcp client more easily.
"""

import numpy as np
from ..streaming.client import Client, Message
from .generic_interface import GenericInterface
from ..enums import (
    DeviceType,
    InterfaceType,
    RealTimeProcessingMethod,
    OfflineProcessingMethod,
    InverseKinematicsMethods,
)
from typing import Union


class TcpClient(GenericInterface):
    """
    Class for interfacing with the client.100
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
        self.read_frequency = read_frequency
        self.ip = ip
        self.port = port
        self.device_cmd_names = []
        self.marker_cmd_names = []
        self.client = Client(server_ip=ip, port=port, client_type=client_type)

    def add_device(
        self,
        nb_channels: int,
        device_type: Union[DeviceType, str] = DeviceType.Emg,
        command_name: Union[str, list] = "",
        name: str = None,
        rate: float = 2000,
        device_range: tuple = None,
        processing_method: Union[RealTimeProcessingMethod, OfflineProcessingMethod] = None,
        **process_kwargs,
    ):
        """
        Add a device to the client.
        Parameters
        ----------
        nb_channels: int
            Number of channels of the device.
        device_type: Union[DeviceType, str]
            Type of the device. (emg, imu, etc.)
        command_name: Union[str, list]
            Name of the command to send to the server.
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
        device_tmp = self._add_device(
            nb_channels, name, device_type, rate, device_range, processing_method, **process_kwargs
        )
        device_tmp.interface = self.interface_type
        self.devices.append(device_tmp)
        self.device_cmd_names.append(command_name)

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
        self.marker_cmd_names.append(subject_name)

    def get_data_from_server(
        self, command: Union[str, list] = "all", nb_frame_to_get: int = None, down_sampling: dict = None
    ):
        """
        Get the data from the server.
        Returns
        -------
        data: Union[str, list]
            ALL the data asked from the server.
        nb_frame_to_get: int
            Number of frame to get.
        down_sampling: dict
            Down sampling parameters. Keys are the names of the devices and values are the down sampling rate.
        """
        all_data = []
        data = self.client.get_data(
            message=Message(command=command, nb_frame_to_get=nb_frame_to_get, down_sampling=down_sampling)
        )
        for key in data:
            if isinstance(data[key], list):
                all_data.append(np.array(data[key]))
            else:
                all_data.append(data[key])
        return all_data

    def get_device_data(
        self, device_name: Union[list, str] = "all", nb_frame_to_get: int = None, down_sampling: dict = None
    ):
        """
        Get the data from a device.

        Parameters
        ----------
        device_name:  Union[list, str]
            Name of the device. all for all the devices.
        nb_frame_to_get: int
            Number of frame to get.
        down_sampling: dict
            Down sampling parameters. Keys are the names of the devices and values are the down sampling rate.

        Returns
        -------
        data: list
            The data asked from the server.
        """
        command = self._prepare_cmd(device_name, True)
        data = self.get_data_from_server(command=command, nb_frame_to_get=nb_frame_to_get, down_sampling=down_sampling)

        for d, device in enumerate(self.devices):
            if device_name == "all" or device.name in device_name:
                if device.name in command:
                    device.new_data = data[command.index(device.name)]
                device.append_data(device.new_data)
        return data

    def get_marker_set_data(
        self, marker_set_name: Union[str, list] = "all", nb_frame_to_get: int = None, down_sampling: dict = None
    ):
        """
        Get the data from the markers.

        Parameters
        ----------
        marker_set_name:  Union[list, str]
            Name of the markers set. all for all the markers sets.
        nb_frame_to_get: int
            Number of frame to get.
        down_sampling: dict
            Down sampling parameters. Keys are the names of the devices and values are the down sampling rate.

        Returns
        -------
        data: list
            The data asked from the server.
        """
        command = self._prepare_cmd(marker_set_name, False)
        data = self.get_data_from_server(command=command, nb_frame_to_get=nb_frame_to_get, down_sampling=down_sampling)
        for m, marker_set in enumerate(self.marker_sets):
            if marker_set_name == "all" or marker_set.name in marker_set_name:
                if marker_set.name in command:
                    marker_set.new_data = data[command.index(marker_set.name)]
                marker_set.append_data(marker_set.new_data)
        return data

    def _prepare_cmd(self, name: Union[str, list], device: bool = True):
        """
        Prepare the command to send to the server.
        Parameters
        ----------
        name: Union[str, list]
            Name of the device or marker set.
        Returns
        -------
        cmd: str
            Command to send to the server.
        """
        if not isinstance(name, list):
            name = [name]

        command = self.device_cmd_names if device else self.marker_cmd_names
        stream_param = self.devices if device else self.marker_sets

        if name != "all":
            command = []
            for s, param in enumerate(stream_param):
                if param.name == name[s]:
                    stream_param.append(device)
                    command.append(command[s])
        return command
