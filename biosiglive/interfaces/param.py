"""
This file is part of biosiglive. It contains the Parameter class and introduce the device and markers classes.
"""
from math import ceil
from ..enums import DeviceType, MarkerType, InverseKinematicsMethods, RealTimeProcessingMethod, OfflineProcessingMethod
from ..processing.data_processing import RealTimeProcessing, OfflineProcessing, GenericProcessing
from ..processing.msk_functions import MskFunctions
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
        self.raw_data = []
        self.processed_data = None
        self.data_windows = data_windows if data_windows else int(rate)
        self.new_data = None

    def _append_data(self, new_data: np.ndarray):
        if len(self.raw_data) == 0:
            self.raw_data = new_data
        elif self.raw_data.shape[len(new_data.shape) - 1] < self.data_windows:
            self.raw_data = np.append(self.raw_data, new_data, axis=len(new_data.shape) - 1)
        else:
            self.raw_data = np.append(
                self.raw_data[..., new_data.shape[len(new_data.shape) - 1] :], new_data, axis=len(new_data.shape) - 1
            )


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
        super().__init__(nb_channels, name, rate, system_rate)
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
        self.process_method = None
        self.process_method_kwargs = {}

    def process(
        self,
        method: Union[str, RealTimeProcessingMethod, OfflineProcessingMethod] = None,
        custom_function: callable = None,
        **kwargs
    ):
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
        method = method if method else self.process_method
        self.process_method_kwargs.update(kwargs)
        if "custom_function" in self.process_method_kwargs.keys():
            custom_function = custom_function if custom_function else self.process_method_kwargs["custom_function"]
        if "processing_windows" in kwargs:
            if self.process_method_kwargs["processing_windows"] != self.data_windows:
                raise ValueError("The processing windows is different than the data windows.")
        if self.new_data is None:
            raise RuntimeError("No data to process. Please run first the function get_device_data.")
        if not method:
            raise RuntimeError("No method to process the data. Please specify a method.")
        if isinstance(method, str):
            if method in [t.value for t in RealTimeProcessingMethod]:
                method = RealTimeProcessingMethod(method)
            elif method not in [t.value for t in OfflineProcessingMethod]:
                method = OfflineProcessingMethod(method)
            else:
                raise ValueError("The method is not valid.")

        if not self.process_method:
            self._init_process_method(method)
        if method == RealTimeProcessingMethod.Custom:
            if not custom_function:
                raise ValueError("No custom function to process the data.")
            self.process_method(custom_function, self.new_data, **self.process_method_kwargs)
        else:
            self.processed_data = self.process_method(self.new_data, **self.process_method_kwargs)
        self._append_data(self.new_data)
        return self.processed_data

    def _init_process_method(self, method: Union[str, RealTimeProcessingMethod, OfflineProcessingMethod]):
        if method == RealTimeProcessingMethod.ProcessEmg:
            self.process_method = RealTimeProcessing(self.rate, self.data_windows).process_emg
        elif method == RealTimeProcessingMethod.ProcessImu:
            self.process_method = RealTimeProcessing(self.rate, self.data_windows).process_imu
        elif method == RealTimeProcessingMethod.GetPeaks:
            self.process_method = RealTimeProcessing(self.rate, self.data_windows).get_peaks
        elif method == OfflineProcessingMethod.ProcessEmg:
            self.process_method = OfflineProcessing(self.rate, self.data_windows).process_emg
        elif method == OfflineProcessingMethod.ComputeMvc:
            self.process_method = OfflineProcessing(self.rate, self.data_windows).compute_mvc
        elif (
            method == RealTimeProcessingMethod.CalibrationMatrix or method == OfflineProcessingMethod.CalibrationMatrix
        ):
            self.process_method = GenericProcessing().calibration_matrix
        elif method == RealTimeProcessingMethod.Custom:
            self.process_method = RealTimeProcessing(self.rate, self.data_windows).custom_processing


class MarkerSet(Param):
    """
    This class is used to store the available markers.
    """

    def __init__(
        self,
        nb_channels: int = 1,
        name: str = None,
        marker_names: Union[str, list] = None,
        rate: float = None,
        unlabeled: bool = False,
        system_rate: float = 100,
    ):
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
        self.kin_method = None
        self.kin_method_kwargs = {}
        self.biorbd_model_path = None

    def get_kinematics(
        self,
        model: callable = None,
        method: Union[InverseKinematicsMethods, str] = None,
        kalman: callable = None,
        custom_function: callable = None,
        **kwargs
    ) -> tuple:
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
            raise RuntimeError(
                "No markers data to compute the kinematics." " Please run first the function get_markers_data."
            )
        method = method if method else self.kin_method
        self.kin_method_kwargs.update(kwargs)
        if "custom_function" in self.kin_method_kwargs.keys():
            custom_function = custom_function if custom_function else self.kin_method_kwargs["custom_function"]
        if self.new_data is None:
            raise RuntimeError("No data to process. Please run first the function get_markers_data.")
        if not method:
            raise RuntimeError("No method to compute the kinematics. Please specify a method.")
        if isinstance(method, str):
            if method in [t.value for t in InverseKinematicsMethods]:
                method = InverseKinematicsMethods(method)
            else:
                raise ValueError("The method is not valid.")
        model = model if model else self.biorbd_model_path
        if model is None:
            raise ValueError("No model to compute the kinematics.")
        msk_class = MskFunctions(model)
        if method == InverseKinematicsMethods.Custom:
            if not custom_function:
                raise ValueError("No custom function to process the data.")
            self.processed_data = msk_class.compute_inverse_kinematics(self.new_data, method, self.rate, custom_function, **self.kin_method_kwargs)
        else:
            self.processed_data = msk_class.compute_inverse_kinematics(self.new_data, method, self.rate, **self.kin_method_kwargs)
        self._append_data(self.new_data)
        return self.processed_data
