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
    def __init__(self, system_rate: float = 100):
        super().__init__(system_rate=system_rate, interface_type=InterfaceType.Custom)

    def add_device(self, nb_channels: int = 1, device_type: Union[DeviceType, str] = DeviceType.Emg, name: str = None,  rate: float = 2000, device_range: tuple = None):
        self.devices.append(self._add_device(nb_channels,device_type,  name,  rate, device_range))

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
            device.new_data = data_tmp
            all_device_data.append(data_tmp)

        if len(all_device_data) == 1:
            return all_device_data[0]
        return all_device_data

    def get_markers_data(self, marker_names: Union[list, str] = None, subject_name: str = None):
        """
        Get the markers data from Vicon.
        Parameters
        ----------
        marker_names: Union[list, str]
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
        if not isinstance(marker_names, list):
            marker_names = [marker_names]
        if not marker_names:
            nb_markers = self.markers[0].nb_channels
        else:
            nb_markers = len(marker_names)
        for m, marker in enumerate(self.markers):
            if marker.name == marker_names[m]:
                marker.marker_data = np.random.rand(3, nb_markers, marker.sample)
                all_markers_data.append(marker.marker_data)
                all_occluded_data.append(occluded)
        if len(all_markers_data) == 1:
            return all_markers_data[0], all_occluded_data[0]
        return all_markers_data, all_occluded_data

    def get_kinematics_from_markers(self, marker_name: str, nb_dof: int):
        self.get_markers_data(marker_name)
        marker_data = None
        for marker in self.markers:
            if marker.name == marker_name:
                marker_data = marker.marker_data
            else:
                raise Exception("No marker with this name")
        return np.random.rand(nb_dof, marker_data.shape[2]), np.random.rand(nb_dof, marker_data.shape[2])


if __name__ == '__main__':
    interface = MyInterface(system_rate=100)
    interface.add_device(nb_channels=8, device_type=DeviceType.Emg, name="My EMG device", rate=2000)
    interface.add_markers(nb_channels=3, name="My markers", marker_names=["M1", "M2", "M3"], rate=100)

    print(interface.get_kinematics_from_markers("My markers", 3))


