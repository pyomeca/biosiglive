import numpy as np
from .generic_interface import GenericInterface
from ..enums import DeviceType, InterfaceType
from typing import Union

try:
    import pytrigno
except ModuleNotFoundError:
    pass


class PytrignoClient(GenericInterface):
    def __init__(self, system_rate=100, ip: str = "127.0.0.1"):
        super(PytrignoClient, self).__init__(ip=ip, interface_type=InterfaceType.PytrignoClient, system_rate=system_rate)
        self.address = ip
        self.devices = []
        self.imu = []
        self.markers = []

        self.emg_client, self.imu_client = None, None
        self.is_frame = False

    def add_device(
        self,
        nb_channels: int,
        device_type: Union[DeviceType, str] = DeviceType.Emg,
        data_buffer_size: int = None,
        name: str = None,
        rate: float = 2000,
        device_range: tuple = None,
    ):
        """
        Add a device to the Pytrigno client.
        Parameters
        ----------
        nb_channels: int
            Number of channels of the device.
        device_type : Union[DeviceType, str]
            Type of the device. (emg or imu)
        name : str
            Name of the device.
        rate : float
            Rate of the device.
        device_range : tuple
            Range of the device.
        """
        device_tmp = self._add_device(nb_channels, device_type, name, rate, device_range)
        device_tmp.interface = self.interface_type
        device_tmp.data_windows = data_buffer_size
        self.devices.append(device_tmp)
        if device_type == DeviceType.Emg:
            self.emg_client = pytrigno.TrignoEMG(
                channel_range=device_tmp.range, samples_per_read=device_tmp.sample, host=self.address
            )
            self.emg_client.start()

        elif device_tmp.type == DeviceType.Imu:
            imu_range = (device_tmp.range[0] * 9, device_tmp.range[1] * 9)
            self.imu_client = pytrigno.TrignoIM(
                channel_range=imu_range, samples_per_read=device_tmp.sample, host=self.address
            )
            self.imu_client.start()

        else:
            raise RuntimeError("Device type must be 'emg' or 'imu' with pytrigno.")

    def get_device_data(self, device_name: str = "all", channel_idx: Union[int, list] = (), get_frame: bool = True):
        """
        Get data from the device.
        Parameters
        ----------
        device_name : str
            Name of the device.
        channel_idx : Union[int, list]
            Index of the channel.
        get_frame : bool
            Get data from device. If False, use the last data acquired.
        Returns
        -------
        data : list
            Data from the device.
        """
        devices = []
        device_data = []
        all_device_data = []
        if not isinstance(channel_idx, list):
            channel_idx = [channel_idx]

        if device_name and not isinstance(device_name, list):
            device_name = [device_name]

        if device_name != "all":
            for d, device in enumerate(self.devices):
                if device.name and device.name == device_name[d]:
                    devices.append(device)
        else:
            devices = self.devices

        for device in devices:
            if get_frame:
                device.new_data = self.emg_client.read() if DeviceType.Emg else self.imu_client.read()
            if len(channel_idx) != 0:
                device_data = np.ndarray((len(channel_idx), device.new_data.shape[1]))
                for i, idx in enumerate(channel_idx):
                    device_data[i, :] = device.new_data[idx, :]
            device_data = device_data if len(channel_idx) != 0 else device.new_data
            if get_frame:
                device.append_data(device.new_data)
            all_device_data.append(device_data)
        return all_device_data

    def get_frame(self):
        self.get_device_data(get_frame=True)
        return True
