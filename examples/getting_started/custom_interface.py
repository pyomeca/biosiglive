from biosiglive.interfaces.generic_interface import GenericInterface
from biosiglive.enums import DeviceType, InterfaceType, InverseKinematicsMethods
from typing import Union
import numpy as np
try:
    import biorbd
    biorbd_installed = True
except ModuleNotFoundError:
    biorbd_installed = False


class MyInterface(GenericInterface):
    def __init__(self):
        super().__init__(system_rate=100, interface_type=InterfaceType.Custom)

    def add_device(self, nb_channels: int = 1, device_type: Union[DeviceType, str] = DeviceType.Emg, name: str = None,  rate: float = 2000, device_range: tuple = (0, 16)):
        self.devices.append(self._add_device(device_type, nb_channels, name,  rate, device_range))

    def add_markers(self,
                    nb_channels: int = 3,
                    name: str = None,
                    marker_names: list = None,
                    rate: float = 100,
                    unlabeled: bool = False,
                    subject_name: str = None,):
        self.markers.append(self._add_markers(nb_channels,
        name,
                                       marker_names,
        rate,
        unlabeled))

    def get_device_data(self, device_name: Union[str, list] = "all", channel_names: str = None):
        """
        Get random data.
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
            data_tmp = np.random.rand(device.nb_channels, device.sample)
            all_device_data.append(data_tmp)

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

        for markers in markers_set:
            markers_data = np.random.rand((3, len(marker_names), markers.sample))
            all_markers_data.append(markers_data)
            all_occluded_data.append(occluded)
        if len(all_markers_data) == 1:
            return all_markers_data[0], all_occluded_data[0]
        return all_markers_data, all_occluded_data

    def get_kinematics_from_markers(self, markers: np.ndarray, model: biorbd.Model, method: Union[InverseKinematicsMethods, str] = InverseKinematicsMethods.BiorbdLeastSquare
                                    , return_qdot: bool=False, custom_func: staticmethod=None, **kwargs):
        if not biorbd_installed:
            raise ModuleNotFoundError("Biorbd is not installed. Please install it to use this function.")
        if return_qdot:
            return np.random.rand(model.nbQ()/2, markers.shape[2]), np.random.rand(model.nbQ()/2, markers.shape[2])
        else:
            return np.random.rand(model.nbQ()/2, markers.shape[2])

