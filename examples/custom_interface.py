from typing import Union
import numpy as np
from biosiglive import (
    GenericInterface,
    DeviceType,
    InterfaceType,
    RealTimeProcessingMethod,
    OfflineProcessingMethod,
    InverseKinematicsMethods,
    load,
)


class MyInterface(GenericInterface):
    def __init__(self, system_rate: float = 100, data_path: str = None):
        super().__init__(system_rate=system_rate, interface_type=InterfaceType.Custom)
        self.offline_data = None
        if data_path:
            self.offline_data = load(data_path)
        self.device_data_key = []
        self.marker_data_key = []
        self.c = 0
        self.d = 0

    def add_device(
        self,
        nb_channels: int = 1,
        device_type: Union[DeviceType, str] = DeviceType.Emg,
        name: str = None,
        data_buffer_size: int = None,
        rate: float = 2000,
        device_range: tuple = None,
        device_data_file_key: str = None,
        processing_method: Union[RealTimeProcessingMethod, OfflineProcessingMethod] = None,
        **process_kwargs,
    ):
        device_tmp = self._add_device(
            nb_channels, device_type, name, rate, device_range, processing_method, **process_kwargs
        )
        device_tmp.data_windows = data_buffer_size
        self.devices.append(device_tmp)
        if self.offline_data is not None:
            if not device_data_file_key:
                raise ValueError("You need to specify the device data file key.")
        self.device_data_key.append(device_data_file_key)

    def add_marker_set(
        self,
        nb_markers: int = 3,
        name: str = None,
        data_buffer_size: int = None,
        marker_names: Union[str, list] = None,
        rate: float = 100,
        unlabeled: bool = False,
        subject_name: str = None,
        unit: str = "m",
        marker_data_file_key: str = None,
        kinematics_method: InverseKinematicsMethods = None,
        **kin_method_kwargs,
    ):
        if len(self.marker_sets) != 0:
            raise ValueError("Only one marker set can be added for now.")

        markers_tmp = self._add_marker_set(
            nb_markers=nb_markers,
            name=name,
            marker_names=marker_names,
            rate=rate,
            unlabeled=unlabeled,
            unit=unit,
            kinematics_method=kinematics_method,
            **kin_method_kwargs,
        )
        if self.offline_data is not None:
            if not marker_data_file_key:
                raise ValueError("You need to specify the marker data file key.")
        markers_tmp.subject_name = subject_name
        markers_tmp.data_windows = data_buffer_size
        self.marker_sets.append(markers_tmp)
        self.marker_data_key.append(marker_data_file_key)

    def get_device_data(self, device_name: Union[str, list] = "all", channel_names: str = None, **kwargs):
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

        for d, device in enumerate(devices):
            if self.offline_data:
                device.new_data = self.offline_data[self.device_data_key[d]][
                    : device.nb_channels, self.c : self.c + device.sample
                ]
                if abs(self.c + device.sample - self.offline_data[self.device_data_key[d]].shape[1]) > device.sample:
                    self.c = self.c + device.sample
                else:
                    self.c = 0
            else:
                device.new_data = np.random.rand(device.nb_channels, device.sample)
            device.append_data(device.new_data)
            all_device_data.append(device.new_data)

        if len(all_device_data) == 1:
            return all_device_data[0]
        return all_device_data

    def get_marker_set_data(self, marker_set_name: Union[list, str] = None, **kwargs):
        """
        Get the markers data from Vicon.
        Parameters
        ----------
        marker_set_name: Union[list, str]
            List of the marker set.
        Returns
        -------
        markers_data: list
            All asked markers data.
        """
        occluded = []
        all_markers_data = []
        all_occluded_data = []
        if not marker_set_name:
            nb_markers = self.marker_sets[0].nb_channels
            marker_set_name = self.marker_sets[0].name
        else:
            nb_markers = len(marker_set_name)
        if not isinstance(marker_set_name, list):
            marker_set_name = [marker_set_name]
        for m, marker in enumerate(self.marker_sets):
            coef = 1 if marker.unit == "m" else 0.001
            if marker.name == marker_set_name[m]:
                if self.offline_data:
                    marker.new_data = (
                        self.offline_data[self.marker_data_key[m]][
                            :, : marker.nb_channels, self.d : self.d + marker.sample
                        ]
                        * coef
                    )
                    if (
                        abs(self.d + marker.sample - self.offline_data[self.marker_data_key[m]].shape[2])
                        > marker.sample
                    ):
                        self.d = self.d + marker.sample
                    else:
                        self.d = 0
                else:
                    marker.new_data = np.random.rand(3, nb_markers, marker.sample)
                marker.append_data(marker.new_data)
                all_markers_data.append(marker.new_data)
                all_occluded_data.append(occluded)
        if len(all_markers_data) == 1:
            return all_markers_data[0], all_occluded_data[0]
        return all_markers_data, all_occluded_data

    def get_kinematics_from_markers(
        self,
        marker_set_name: str,
        nb_dof: int = None,
        model_path: str = None,
        method: Union[InverseKinematicsMethods, str] = InverseKinematicsMethods.BiorbdLeastSquare,
        custom_func: callable = None,
        get_markers_data: bool = False,
        **kwargs,
    ):
        if get_markers_data:
            self.get_marker_set_data(marker_set_name)
        if not self.offline_data:
            if not nb_dof:
                raise Exception("You need to specify the number of dof")
            return np.random.rand(nb_dof, 1), np.random.rand(nb_dof, 1)
        else:
            marker_set_idx = [i for i, m in enumerate(self.marker_sets) if m.name == marker_set_name][0]
            return self.marker_sets[marker_set_idx].get_kinematics(
                model_path, method, custom_func=custom_func, **kwargs
            )


if __name__ == "__main__":
    interface = MyInterface(system_rate=100, data_path="abd.bio")
    interface.add_device(
        nb_channels=8, device_type=DeviceType.Emg, name="My EMG device", rate=2000, device_data_file_key="emg"
    )
    interface.add_marker_set(
        nb_channels=15,
        name="My markers",
        rate=100,
        data_buffer_size=100,
        marker_data_file_key="markers",
        model_path="model/Wu_Shoulder_Model_mod_wt_wrapp.bioMod",
        kinematics_method=InverseKinematicsMethods.BiorbdKalman,
    )

    print(interface.get_kinematics_from_markers("My markers", get_markers_data=True))
