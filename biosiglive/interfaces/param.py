"""
This file is part of biosiglive. It contains the Parameter class and introduce the device and markers classes.
"""
from math import ceil
from ..processing.data_processing import RealTimeProcessing, OfflineProcessing, GenericProcessing


class Type:
    def __init__(self, name: str = None, type: str = None, rate: float = None, system_rate: float = 100, real_time: bool = False):
        """
        initialize the parameter class

        Parameters
        ----------
        name : str
            name of the parameter
        type : str
            type of the parameter (emg, imu, markers, ...)
        rate : float
            rate of the parameter
        system_rate : float
            rate of the system
        real_time : bool
            If true device wil be use in real time application
        """

        self.name = name
        self.type = type
        self.rate = rate
        self.system_rate = system_rate
        self.sample = ceil(rate / self.system_rate)
        self.range = None
        self.process_method = None
        # self._set_process_method(real_time)

    def set_name(self, name: str):
        self.name = name

    def set_type(self, type: str):
        self.type = type

    def set_rate(self, rate: int):
        self.rate = rate

    def get_process_method(self):
        return self.process_method

    def set_process_method(self, processing_class):
        self.process_method = processing_class


class Device(Type):
    """
    This class is used to store the available devices.
    """

    def __init__(self, name: str = None, type: str = "emg", rate: float = 2000, system_rate: float = 100, channel_names: list = None):
        super().__init__(name, type, rate, system_rate)
        self.infos = None
        self.channel_names = channel_names

    def add_channel_names(self, channel_names: list):
        """
        add the channel names to the device
        Parameters
        ----------
        channel_names: list
            list of channel names
        """
        self.channel_names = channel_names


# class Imu(Type):
#     """
#     This class is used to store the available IMU devices.
#     """
#     def __init__(self, name: str = None, rate: float = None, from_emg: bool = False):
#         type = "imu" if not from_emg else "imu_from_emg"
#         super().__init__(name, type, rate)


class MarkerSet(Type):
    """
    This class is used to store the available markers.
    """
    def __init__(self, name: str = None, rate: float = None, unlabeled: bool = False):
        type = "unlabeled" if unlabeled else "labeled"
        super().__init__(name, type, rate)
        self.markers_names = name
        self.subject_name = None
