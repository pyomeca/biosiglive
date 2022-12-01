"""
This file is part of biosiglive. It contains a wrapper for the Vicon SDK for Python.
"""
from .param import *
from typing import Union
from ..enums import DeviceType, InverseKinematicsMethods, InterfaceType


class GenericInterface:
    """
    Class for generic interfacing.
    """

    def __init__(
        self, ip: str = "127.0.0.1", system_rate: float = 100, interface_type: Union[InterfaceType, str] = None
    ):
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
        self.marker_sets = []
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
        processing_method: Union[RealTimeProcessingMethod, OfflineProcessingMethod] = None,
        **kwargs,
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
        if name in [device.name for device in self.devices]:
            raise RuntimeError(f"The device '{name}' already exists.")
        if isinstance(device_type, str):
            if device_type not in [t.value for t in DeviceType]:
                raise ValueError("The type of the device is not valid.")
            device_type = DeviceType(device_type)
        device_tmp = Device(device_type, nb_channels, name, rate, self.system_rate)
        device_tmp.device_range = device_range if device_range else (0, nb_channels)
        device_tmp.interface = self.interface_type
        device_tmp.processing_method = processing_method
        device_tmp.processing_method_kwargs = kwargs
        return device_tmp

    def _add_marker_set(
        self,
        nb_markers: int,
        name: str = None,
        marker_names: Union[str, list] = None,
        rate: float = 100,
        unlabeled: bool = False,
        unit: str = "m",
        kinematics_method: Union[InverseKinematicsMethods, str] = None,
        **kwargs,
    ):
        """
        Add a marker set to stream from the Vicon system.
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
        if name in [marker.name for marker in self.marker_sets]:
            raise RuntimeError(f"The marker set '{name}' already exists.")
        markers_tmp = MarkerSet(nb_markers, name, marker_names, rate, unlabeled, self.system_rate)
        markers_tmp.kin_method = kinematics_method
        markers_tmp.kin_method_kwargs = kwargs
        markers_tmp.interface = self.interface_type
        markers_tmp.unit = unit
        return markers_tmp

    def add_marker_set(self, **kwargs):

        """
        Add a marker set to stream from the interface system.
        """
        raise RuntimeError(f"Markers are not implemented with interface '{self.interface_type}'.")

    def get_device(self, name: str = None, idx: int = None):
        """
        Get a device from the interface.
        Parameters
        ----------
        idx: int
            Index of the device.
        name: str
            Name of the device.
        Returns
        -------
        The device.
        """
        if idx is not None:
            return self.devices[idx]
        elif name is not None:
            for device in self.devices:
                if device.name == name:
                    return device
        else:
            raise RuntimeError("You must provide an index or a name for the device.")

    def get_marker_set(self, name: str = None, idx: int = None):
        """
        Get a device from the interface.
        Parameters
        ----------
        idx: int
            Index of the device.
        name: str
            Name of the device.
        Returns
        -------
        The device.
        """
        if idx is not None:
            return self.marker_sets[idx]
        elif name is not None:
            for marker_set in self.marker_sets:
                if marker_set.name == name:
                    return marker_set
        else:
            raise RuntimeError("You must provide an index or a name for the marker set.")

    def add_device(self, **kwargs):
        """
        Add a device to the interface.
        """
        raise RuntimeError(f"Devices are not implemented with interface '{self.interface_type}.")

    @staticmethod
    def get_force_plate_data():
        raise NotImplementedError("Force plate streaming is not implemented yet.")

    def get_device_data(self, **kwargs):
        raise RuntimeError(f"You can not get device data from '{self.interface_type}'.")

    def get_marker_set_data(self, **kwargs):
        raise RuntimeError(f"You can not get markers data from '{self.interface_type}'.")

    def get_latency(self):
        return -1

    def get_frame(self):
        return True

    def get_frame_number(self):
        raise RuntimeError(f"You can not get frame number from '{self.interface_type}'.")

    def get_kinematics_from_marker_set(self, **kwargs):
        raise RuntimeError(f"You can not get kinematics from '{self.interface_type}'.")

    def init_client(self):
        raise RuntimeError(f"You can not init client from '{self.interface_type}'.")
