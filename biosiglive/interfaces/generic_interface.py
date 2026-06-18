"""
This file contains a generic interface class to use for any new implemented class.
"""

from .param import *
from typing import Union
from ..enums import DeviceType, InverseKinematicsMethods, InterfaceType


class GenericInterface:
    """
    Class for generic interfacing.
    """

    def __init__(
        self, ip: str = "127.0.0.1", system_rate: float = None, interface_type: Union[InterfaceType, str] = None
    ):
        """
        Initialize the generic class.

        Parameters
        ----------
        ip: str
            IP address of the interface.
        system_rate: float
            Rate of the system which record the data.
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
        rate: float = None,
        processing_method: Union[RealTimeProcessingMethod, OfflineProcessingMethod] = None,
        **kwargs,
    ):
        """
        Add a device to the interface.

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
        processing_method: Union[RealTimeProcessingMethod, OfflineProcessingMethod]
            Processing method to use to process the device.
        kwargs:
            Keyword arguments for the processing method.
        """
        if name in [device.name for device in self.devices]:
            raise RuntimeError(f"The device '{name}' already exists.")
        if isinstance(device_type, str):
            if device_type not in [t.value for t in DeviceType]:
                raise ValueError("The type of the device is not valid.")
            device_type = DeviceType(device_type)
        device_tmp = Device(device_type, nb_channels, name, rate, self.system_rate)
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
        Add a marker set to stream from the interface.

        Parameters
        ----------
        nb_markers: int
            Number of markers in the marker set.
        name: str
            Name of the markers set.
        marker_names: Union[list, str]
            List of markers names.
        rate: int
            Rate of the markers set.
        unlabeled: bool
            Whether the markers set is unlabeled.
        unit: str
            Unit of the markers set.
        kinematics_method: Union[InverseKinematicsMethods, str]
            Inverse kinematics method to use to process the markers set.
        kwargs:
            Keyword arguments for the inverse kinematics method.
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
        The device object.
        """
        if idx is not None:
            if name:
                raise RuntimeError("You cannot provide an index and a name for the device.")
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
        name: str
            Name of the device.
        idx: int
            Index of the device.

        Returns
        -------
        The device.
        """
        if idx is not None:
            if name:
                raise RuntimeError("You cannot provide an index and a name for the device.")
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
        """
        Get the force plate data.
        """
        raise NotImplementedError("Force plate streaming is not implemented yet.")

    def get_device_data(self, **kwargs):
        """
        Get the device data.
        """
        raise RuntimeError(f"You can not get device data from '{self.interface_type}'.")

    def get_marker_set_data(self, **kwargs):
        """
        Get the marker set data.
        """
        raise RuntimeError(f"You can not get markers data from '{self.interface_type}'.")

    def get_latency(self):
        """
        Get the latency of the interface.
        """
        return -1

    def get_frame(self):
        """
        Get the frame of the interface. That need to be call to retrieve all the data once at the same time.
        """
        return True

    def get_frame_number(self):
        """
        Get the frame number of the interface.
        """
        raise RuntimeError(f"You can not get frame number from '{self.interface_type}'.")

    def get_kinematics_from_marker_set(self, **kwargs):
        """
        Get the kinematics from a marker set.
        """
        raise RuntimeError(f"You can not get kinematics from '{self.interface_type}'.")

    def init_client(self):
        """
        Initialize the client.
        """
        raise RuntimeError(f"You can not init client from '{self.interface_type}'.")
