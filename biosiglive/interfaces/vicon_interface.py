"""
This file is part of biosiglive. It contains a wrapper for the Vicon SDK for Python.
"""

import numpy as np
from .param import *
from typing import Union
from .generic_interface import GenericInterface
from ..enums import InverseKinematicsMethods, InterfaceType
from ..processing.msk_functions import compute_inverse_kinematics

try:
    from vicon_dssdk import ViconDataStream as VDS
except ModuleNotFoundError:
    pass


class ViconClient(GenericInterface):
    """
    Class for interfacing with the Vicon system.
    """

    def __init__(self, system_rate: int, ip: str = "127.0.0.1", port: int = 801, init_now=True):
        """
        Initialize the ViconClient class.
        Parameters
        ----------
        system_rate: int
            Streaming rate of the nexus software.
        ip: str
            IP address of the Vicon system.
        port: int
            Port of the Vicon system.
        """
        super(ViconClient, self).__init__(ip=ip, system_rate=system_rate, interface_type=InterfaceType.ViconClient)
        self.address = f"{ip}:{port}"

        self.vicon_client = None
        self.acquisition_rate = None
        self.system_rate = system_rate

        if init_now:
            self._init_client()

        self.devices = []
        self.imu = []
        self.markers = []
        self.is_frame = False
        self.kalman = None

    def _init_client(self):
        print(f"Connection to ViconDataStreamSDK at : {self.address} ...")
        self.vicon_client = VDS.Client()
        self.vicon_client.Connect(self.address)
        print("Connected to Vicon.")

        # Enable some different data types
        self.vicon_client.EnableSegmentData()
        self.vicon_client.EnableDeviceData()
        self.vicon_client.EnableMarkerData()
        self.vicon_client.EnableUnlabeledMarkerData()
        self.get_frame()

    def add_device(self, name: str, device_type: Union[DeviceType, str] = DeviceType.Emg, rate: float = 2000, device_range: tuple = (0, 16)):
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
        device_tmp = self._add_device(name, device_type, rate, device_range)
        device_tmp.interface = self.interface_type
        if self.vicon_client:
            device_tmp.info = self.vicon_client.GetDeviceOutputDetails(name)
        else:
            device_tmp.info = None

        self.devices.append(device_tmp)

    def add_markers(
        self,
        name: str = None,
        marker_names : Union[list, str] = None,
        rate: float = 100,
        unlabeled: bool = False,
        subject_name: str = None,
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
        subject_name: str
            Name of the subject. If None, the subject will be the first one in Nexus.
        """
        if len(self.markers) != 0:
            raise ValueError("Only one marker set can be added for now.")
        markers_tmp = MarkerSet(name, rate, unlabeled)
        markers_tmp.interface = self.interface_type
        if self.vicon_client:
            markers_tmp.subject_name = subject_name if subject_name else self.vicon_client.GetSubjectNames()[0]
            markers_tmp.markers_names = self.vicon_client.GetMarkerNames(markers_tmp.subject_name) if not name else name
        else:
            markers_tmp.subject_name = subject_name
            markers_tmp.markers_names = marker_names
        self.markers.append(markers_tmp)

    @staticmethod
    def get_force_plate_data(vicon_client):
        raise NotImplementedError("Force plate streaming is not implemented yet.")

    def get_device_data(self, device_name: Union[str, list] = "all", channel_names: str = None):
        """
        Get the device data from Vicon.
        Parameters
        ----------
        device_name: str or list
            Name of the device or list of devices names.
        channel_names: str
            Name of the channel.

        Returns
        -------
        device_data: list
            All asked device data.
        """
        devices = []
        all_device_data = []
        if not isinstance(device_name, list):
            device_name = [device_name]

        if device_name != ["all"]:
            for d, device in enumerate(self.devices):
                if device.name == device_name[d]:
                    devices.append(device)
        else:
            devices = self.devices

        for device in devices:
            if not device.infos:
                device.infos = self.vicon_client.GetDeviceOutputDetails(device.name)

            if channel_names:
                device_data = np.zeros((len(channel_names), device.sample))
            else:
                device_data = np.zeros((len(device.infos), device.sample))

            count = 0
            device_chanel_names = []
            for output_name, chanel_name, unit in device.infos:
                data_tmp, _ = self.vicon_client.GetDeviceOutputValues(device.name, output_name, chanel_name)
                if channel_names:
                    if chanel_name in channel_names:
                        device_data[count, :] = data_tmp
                        device_chanel_names.append(chanel_name)
                else:
                    device_data[count, :] = data_tmp
                    device_chanel_names.append(chanel_name)
                device.chanel_names = device_chanel_names
                count += 1
            all_device_data.append(device_data)
        if len(all_device_data) == 1:
            return all_device_data[0]
        return all_device_data

    def get_markers_data(self, marker_names: list = None, subject_name: str = None):
        """
        Get the markers data from Vicon.
        Parameters
        ----------
        marker_names: list
            List of markers names.
        subject_name: str
            Name of the subject. If None, the subject will be the first one in Nexus.

        Returns
        -------
        markers_data: list
            All asked markers data.
        """
        markers_set = []
        occluded = []
        all_markers_data = []
        all_occluded_data = []
        if subject_name:
            for s, marker_set in enumerate(self.markers):
                if marker_set.subject_name == subject_name[s]:
                    markers_set.append(marker_set)
                    marker_names_tmp = self.vicon_client.GetMarkerNames(marker_set.subject_name)
                    markers_set[-1].markers_names = []
                    for i in range(len(marker_names_tmp)):
                        markers_set[-1].markers_names.append(marker_names_tmp[i][0])
        else:
            for i in range(len(self.markers)):
                marker_names_tmp = self.vicon_client.GetMarkerNames(self.markers[i].subject_name)
                self.markers[i].markers_names = []
                for j in range(len(marker_names_tmp)):
                    self.markers[i].markers_names.append(marker_names_tmp[j][0])
            markers_set = self.markers

        for markers in markers_set:
            markers_data = np.zeros((3, len(markers.markers_names), markers.sample))
            count = 0
            for m, marker_name in enumerate(markers.markers_names):
                markers_data_tmp, occluded_tmp = self.vicon_client.GetMarkerGlobalTranslation(
                    markers.subject_name, marker_name
                )
                if marker_names:
                    if marker_name in marker_names:
                        markers_data[:, count, :] = markers_data_tmp
                        occluded.append(occluded_tmp)
                else:
                    markers_data[:, count, :] = np.array(markers_data_tmp)[:, np.newaxis] * 0.001
                    occluded.append(occluded_tmp)
                count += 1
            all_markers_data.append(markers_data)
            all_occluded_data.append(occluded)
        if len(all_markers_data) == 1:
            return all_markers_data[0], all_occluded_data[0]
        return all_markers_data, all_occluded_data

    def get_latency(self):
        return self.vicon_client.GetLatencyTotal()

    def get_frame(self):
        self.is_frame = self.vicon_client.GetFrame()
        while self.is_frame is not True:
            self.is_frame = self.vicon_client.GetFrame()

    def get_frame_number(self):
        return self.vicon_client.GetFrameNumber()

    def get_kinematics_from_markers(self, markers: np.ndarray, model: biorbd.Model, method: Union[InverseKinematicsMethods, str] =InverseKinematicsMethods.BiorbdLeastSquare
                                    , return_qdot: bool=False, custom_func: staticmethod=None, **kwargs):
        """
        Get the kinematics from markers.
        Parameters
        ----------
        markers: np.ndarray
            Array of markers.
        model: biorbd.Model
            Model of the kinematics.
        method: str
            Method to use to get the kinematics. Can be "kalman" or "custom".
        return_qdot: bool
            If True, return the qdot.
        custom_func: function
            Custom function to get the kinematics.
        Returns
        -------
        kinematics: list
            List of kinematics.
        """
        if isinstance(method, str):
            if method in [t.value for t in InverseKinematicsMethods]:
                method = InverseKinematicsMethods(method)
            else:
                raise ValueError(f"Method {method} is not supported")

        if method == InverseKinematicsMethods.BiorbdKalman:
            q, q_dot, self.kalman = compute_inverse_kinematics(markers, model, return_q_dot=True, use_kalman=True, kalman=self.kalman)
        elif method == InverseKinematicsMethods.BiorbdLeastSquare:
            q, q_dot = compute_inverse_kinematics(markers, model, return_q_dot=True, use_kalman=False)
        elif method == InverseKinematicsMethods.Custom:
            q, q_dot = custom_func(markers, model, **kwargs)
        else:
            raise RuntimeError(f"Method f{method} not implemented")

        if return_qdot:
            return q, q_dot
        else:
            return q