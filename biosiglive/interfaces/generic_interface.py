"""
This file is part of biosiglive. It contains a wrapper for the Vicon SDK for Python.
"""

import numpy as np
from .param import *
from typing import Union
from ..enums import DeviceType, InverseKinematicsMethods, InterfaceType
try:
    import biorbd
except ModuleNotFoundError:
    pass


class GenericInterface:
    """
    Class for generic interfacing.
    """

    def __init__(self, ip: str = "127.0.0.1", system_rate: float = 100, interface_type: Union[InterfaceType, str] = None):
        """
        Initialize the ViconClient class.
        Parameters
        ----------
        ip: str
            IP address of the interface.
        system_rate: float
            Rate of the system.
        interface_type: Union[InterfaceType, str]
            Type of the interface.
        """
        self.ip = ip
        self.system_rate = system_rate
        self.acquisition_rate = None
        self.devices = []
        self.imu = []
        self.markers = []
        self.is_frame = False
        self.kalman = None
        if isinstance(interface_type, str):
            if interface_type not in [t.value for t in InterfaceType]:
                raise ValueError("The type of interface is not valid.")
            self.interface_type = InterfaceType(interface_type)
        else:
            self.interface_type = interface_type

    def _add_device(
        self,
        nb_channels: int,
        device_type: Union[DeviceType, str] = DeviceType.Emg,
        name: str = None,
        rate: float = 2000,
        device_range: tuple = None,
    ):
        """
        Add a device to the Vicon system.
        Parameters
        ----------
        name: str
            Name of the device.
        device_type: Union[DeviceType, str]
            Type of the device.
        rate: float
            Rate of the device.
        device_range: tuple
            Range of the device.
        """
        if isinstance(device_type, str):
            if device_type not in [t.value for t in DeviceType]:
                raise ValueError("The type of the device is not valid.")
            device_type = DeviceType(device_type)
        device_tmp = Device(device_type, nb_channels,  name,  rate, self.system_rate)
        device_tmp.device_range = device_range if device_range else (0, nb_channels)
        device_tmp.interface = self.interface_type
        return device_tmp

    def _add_markers(
        self,
        nb_markers: int,
        name: str = None,
        marker_names: Union[str, list] = None,
        rate: float = 100,
        unlabeled: bool = False,
    ):
        """
        Add markers set to stream from the Vicon system.
        Parameters
        ----------
        name: str
            Name of the markers set.
        marker_names: Union[list, str]
            List of markers names.
        rate: int
            Rate of the markers set.
        unlabeled: bool
            Whether the markers set is unlabeled.
        """
        markers_tmp = MarkerSet(nb_markers, name, marker_names, rate, unlabeled, self.system_rate)
        markers_tmp.interface = self.interface_type
        return markers_tmp

    def add_markers(self, **kwargs):

        """
        Add markers set to stream from the interface system.
        """
        raise RuntimeError(f"Markers are not implemented with interface '{self.interface_type}'.")

    def add_device(self, **kwargs):
        """
        Add a device to the Vicon system.
        """
        raise RuntimeError(f"Devices are not implemented with interface '{self.interface_type}.")

    @staticmethod
    def get_force_plate_data():
        raise NotImplementedError("Force plate streaming is not implemented yet.")

    def get_device_data(self, **kwargs):
        raise RuntimeError(f"You can not get device data from '{self.interface_type}'.")

    def get_markers_data(self, **kwargs):
        raise RuntimeError(f"You can not get merkers data from '{self.interface_type}'.")

    def get_latency(self):
        raise RuntimeError(f"You can not get latency from '{self.interface_type}'.")

    def get_frame(self):
        raise RuntimeError(f"You can not get frames from '{self.interface_type}'.")

    def get_frame_number(self):
        raise RuntimeError(f"You can not get frame number from '{self.interface_type}'.")

    def get_kinematics_from_markers(self, **kwargs):
        raise RuntimeError(f"You can not get kinematics from '{self.interface_type}'.")

    def init_client(self):
        raise RuntimeError(f"You can not init client from '{self.interface_type}'.")

