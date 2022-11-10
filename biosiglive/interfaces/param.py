"""
This file is part of biosiglive. It contains the Parameter class and introduce the device and markers classes.
"""
from math import ceil
from ..enums import DeviceType, MarkerType, InverseKinematicsMethods, RealTimeProcessingMethod, OfflineProcessingMethod
from ..processing.data_processing import RealTimeProcessing, OfflineProcessing, GenericProcessing
from ..processing.msk_functions import compute_inverse_kinematics
from typing import Union
try:
    import biorbd
except ModuleNotFoundError:
    pass
import numpy as np


class Param:
    def __init__(
        self,
        nb_channels: int,
        name: str = None,
        rate: float = None,
        system_rate: float = 100,
        data_windows: int = None,
    ):
        """
        initialize the parameter class

        Parameters
        ----------
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
        self.rate = rate
        self.system_rate = system_rate
        self.sample = ceil(rate / self.system_rate)
        self.range = None
        self.process_method = None
        self.raw_data = []
        self.processed_data = None
        self.data_windows = data_windows if data_windows else int(rate)
        self.new_data = None

    def _append_data(self, new_data: np.ndarray):
        if len(self.raw_data) == 0:
            self.raw_data = new_data
        elif self.raw_data.shape[len(new_data.shape)-1] < self.data_windows:
            self.raw_data = np.append(self.raw_data, new_data, axis=len(new_data.shape)-1)
        else:
            self.raw_data = np.append(self.raw_data[..., new_data.shape[len(new_data.shape)-1]:], new_data, axis=len(new_data.shape)-1)

    def set_name(self, name: str):
        self.name = name

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
        super().__init__(nb_channels, name,  rate, system_rate)
        if isinstance(channel_names, str):
            channel_names = [channel_names]
        if channel_names:
            if nb_channels != len(channel_names):
                raise ValueError("The number of channels is not equal to the number of channel names.")
        self.device_range = None
        self.infos = None
        self.channel_names = channel_names
        self.interface = None
        self.device_type = device_type

    def process(self, method: Union[str, RealTimeProcessingMethod, OfflineProcessingMethod], custom_function: callable = None, **kwargs):
        """
        Process the data of the device.
        Parameters
        ----------
        method: callable
            Method to process the data.
        custom_function: callable
            Custom function to process the data.
        kwargs: dict
            Keyword arguments to pass to the method.
        """
        if "processing_windows" in kwargs:
            if kwargs["processing_windows"] != self.data_windows:
                raise ValueError("The processing windows is different than the data windows.")
            kwargs.pop("moving_average_window")

        if "moving_average_window" in kwargs:
            ma_win = kwargs["moving_average_window"]
            kwargs.pop("moving_average_window")
        else:
            ma_win = ceil(self.rate/10)
        if self.new_data is None:
            raise RuntimeError("No data to process. Please run first the function get_device_data.")
        if isinstance(method, str):
            if method in [t.value for t in RealTimeProcessingMethod]:
                method = RealTimeProcessingMethod(method)
            elif method not in [t.value for t in OfflineProcessingMethod]:
                method = RealTimeProcessingMethod(method)

        if not self.process_method:
            self._init_process_method(method, **kwargs)

        self.processed_data = self.process_method(self.new_data, **kwargs)
        self._append_data(self.new_data)
        return self.processed_data

    def _init_process_method(self,  method: Union[str, RealTimeProcessingMethod, OfflineProcessingMethod], **kwargs):
        if "processing_windows" in kwargs:
            if kwargs["processing_windows"] != self.data_windows:
                raise ValueError("The processing windows is different than the data windows.")
            kwargs.pop("moving_average_window")

        if "moving_average_window" in kwargs:
            ma_win = kwargs["moving_average_window"]
            kwargs.pop("moving_average_window")
        else:
            ma_win = ceil(self.rate / 10)
        if method == RealTimeProcessingMethod.ProcessEmg:
            self.process_method = RealTimeProcessing(self.rate, self.data_windows, ma_win).process_emg
        elif method == RealTimeProcessingMethod.ProcessImu:
            self.process_method = RealTimeProcessing(self.rate, self.data_windows, ma_win).process_imu
        elif method == RealTimeProcessingMethod.GetPeaks:
            self.process_method = RealTimeProcessing(self.rate, self.data_windows, ma_win).get_peaks
        elif method == OfflineProcessingMethod.ProcessEmg:
            self.process_method = OfflineProcessing(self.rate, self.data_windows, ma_win).process_emg
        elif method == OfflineProcessingMethod.ComputeMvc:
            self.process_method = OfflineProcessing(self.rate, self.data_windows, ma_win).compute_mvc
        elif method == RealTimeProcessingMethod.CalibrationMatrix or method == OfflineProcessingMethod.CalibrationMatrix:
            self.process_method = GenericProcessing().calibration_matrix
        elif method == RealTimeProcessingMethod.Custom:
            self.process_method = RealTimeProcessing(self.rate, self.data_windows, ma_win).custom_processing
        else:
            raise ValueError("The method is not a valid method.")



class MarkerSet(Param):
    """
    This class is used to store the available markers.
    """

    def __init__(self, nb_channels: int = 1, name: str = None, marker_names: Union[str, list] = None, rate: float = None, unlabeled: bool = False, system_rate: float = 100):
        marker_type = MarkerType.Unlabeled if unlabeled else MarkerType.Labeled
        super(MarkerSet, self).__init__(nb_channels, name, rate, system_rate)
        if isinstance(marker_names, str):
            marker_names = [marker_names]
        if marker_names:
            if nb_channels != len(marker_names):
                raise ValueError("The number of channels and the number of markers names are not the same.")
        self.marker_names = marker_names
        self.subject_name = None
        self.interface = None
        self.marker_type = marker_type

    def get_kinematics(self, model: callable, method: Union[InverseKinematicsMethods, str] = InverseKinematicsMethods.BiorbdLeastSquare,
                       kalman: callable=None, custom_function: callable=None, **kwargs)->tuple:
        """
        Function to apply the Kalman filter to the markers.
        Parameters
        ----------
        model : biorbd.Model
            The model used to compute the kinematics.
        kalman : biorbd.KalmanReconsMarkers
            The Kalman filter to use.
        method : Union[InverseKinematicsMethods, str]
            The method to use to compute the inverse kinematics.
        custom_function : callable
            Custom function to use.
        Returns
        -------
        tuple
            The joint angle and velocity.
        """
        if not self.raw_data:
            raise RuntimeError("No markers data to compute the kinematics."
                               " Please run first the function get_markers_data.")
        return compute_inverse_kinematics(self.raw_data, model, method, self.rate, kalman, custom_function, **kwargs)
