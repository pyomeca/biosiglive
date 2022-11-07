"""
This file is part of biosiglive. It contains the Parameter class and introduce the device and markers classes.
"""
from math import ceil
from ..enums import DeviceType, MarkerType
from typing import Union


class Param:
    def __init__(
        self,
        param_type: Union[DeviceType, MarkerType],
        nb_channels: int,
        name: str = None,
        rate: float = None,
        system_rate: float = 100,
    ):
        """
        initialize the parameter class

        Parameters
        ----------
        param_type : Union[DeviceType, MarkerType]
            enum type of the parameter (emg, imu, markers, ...)
        nb_channels : int
            number of channels of the parameter
        name : str
            name of the parameter
        rate : float
            rate of the parameter
        system_rate : float
            rate of the system
        """
        self.nb_channels = nb_channels
        self.name = name
        self.param_type = param_type
        self.rate = rate
        self.system_rate = system_rate
        self.sample = ceil(rate / self.system_rate)
        self.range = None
        self.process_method = None

    def set_name(self, name: str):
        self.name = name

    def set_type(self, param_type: Union[DeviceType, MarkerType]):
        self.param_type = param_type

    def get_type(self):
        return self.param_type

    def set_rate(self, rate: int):
        self.rate = rate

    def get_process_method(self):
        return self.process_method

    def set_process_method(self, processing_class):
        self.process_method = processing_class


class Device(Param):
    """
    This class is used to store the available devices.
    """

    def __init__(
        self,
        device_type: DeviceType = DeviceType.Emg,
        nb_channels: int = 1,
        name: str = None,
        rate: float = 2000,
        system_rate: float = 100,
        channel_names: Union[list, str] = None,
    ):
        super().__init__(device_type, nb_channels, name,  rate, system_rate)
        if isinstance(channel_names, str):
            channel_names = [channel_names]
        if nb_channels != len(channel_names):
            raise ValueError("The number of channels is not equal to the number of channel names.")
        self.device_range = None
        self.infos = None
        self.channel_names = channel_names
        self.interface = None

    def add_channel_names(self, channel_names: list):
        """
        add the channel names to the device
        Parameters
        ----------
        channel_names: list
            list of channel names
        """
        self.channel_names = channel_names


class MarkerSet(Param):
    """
    This class is used to store the available markers.
    """

    def __init__(self, nb_channels: int = 1, name: str = None, marker_names: Union[str, list] = None, rate: float = None, unlabeled: bool = False, system_rate: float = 100):
        marker_type = MarkerType.Unlabeled if unlabeled else MarkerType.Labeled
        super().__init__(marker_type, nb_channels, name, rate, system_rate)
        if isinstance(marker_names, str):
            self.marker_names = [marker_names]
        if nb_channels != len(self.marker_names):
            raise ValueError("The number of channels and the number of markers names are not the same.")
        self.markers_names = marker_names
        self.subject_name = None
        self.interface = None
