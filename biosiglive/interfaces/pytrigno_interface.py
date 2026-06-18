import numpy as np
from .generic_interface import GenericInterface
from ..enums import DeviceType, InterfaceType, RealTimeProcessingMethod, OfflineProcessingMethod
from typing import Union

# try:
#     import pytrigno
# except ModuleNotFoundError:
#     pass
from .trigno_sdk.sdk_client import TrignoSDKClient


class PytrignoClient(GenericInterface):
    """
    Class to wrap the Trigno community SDK.
    """

    def __init__(self, system_rate=74.074074, ip: str = "127.0.0.1", init_now: bool = True):
        """
        Initialize the interface.

        Parameters
        ----------
        system_rate: int
            Rate of the system.
        ip: str
            IP address of the Trigno system.
        init_now: bool
            Initialize the client the same time that it is created.
        """
        super(PytrignoClient, self).__init__(
            ip=ip, interface_type=InterfaceType.PytrignoClient, system_rate=system_rate
        )
        self.address = ip
        self.devices = []
        self.imu = []
        self.markers = []

        self.emg_client, self.imu_client = [], []
        self.is_frame = False
        self.is_initialized = False
        self.init_now = init_now
        self.sdk_client = None
        # if system_rate != 74.074074:
        #     raise ValueError("System rate can not be changed for pytrigno interface for now." \
        #     " 74 Hz mean that data will refresh every 13.5 ms but the EMG and IMU data are sampled at their own rate.")
        if self.init_now:
            self.sdk_client = TrignoSDKClient(host=ip, init_sensors=False, stream_rate=system_rate)

    def add_device(
        self,
        nb_channels: int,
        device_type: Union[DeviceType, str] = DeviceType.Emg,
        data_buffer_size: int = None,
        name: str = None,
        rate: float = 2000,
        processing_method: Union[RealTimeProcessingMethod, OfflineProcessingMethod] = None,
        **process_kwargs,
    ):
        """
        Add a device to the Pytrigno system.

        Parameters
        ----------
        nb_channels: int
            Number of channels of the device will be remove in futur version.
        device_type: Union[DeviceType, str]
            Type of the device.
        data_buffer_size: int
            Size of the buffer for the device.
        name: str
            Name of the device.
        rate: float
            Rate of the device.
        device_range: tuple
            Range of the device. Number of selected channels. If None, all channels are selected (0, 16).
        processing_method : Union[RealTimeProcessingMethod, OfflineProcessingMethod]
            Method used to process the data.
        **process_kwargs
            Keyword arguments for the processing method.
        """
        device_tmp = self._add_device(nb_channels, device_type, name, rate, processing_method, **process_kwargs)
        device_tmp.interface = self.interface_type
        device_tmp.data_windows = data_buffer_size
        self.devices.append(device_tmp)
        if isinstance(device_type, str):
            if device_type not in [t.value for t in DeviceType]:
                raise ValueError("The type of the device is not valid.")
            device_type = DeviceType(device_type)
        if (
            device_type != DeviceType.Emg
            and device_type != DeviceType.Imu
            and device_type != DeviceType.DelsysGogniometer
        ):
            raise RuntimeError("Device type must be 'emg', 'delsys_gogniometer' or 'imu' with pytrigno.")

        if self.init_now:
            self.init_client()

    def get_device_data(
        self, device_name: str = "all", channel_idx: Union[int, list] = (), get_frame: bool = True
    ) -> np.ndarray:
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
        if not self.is_initialized:
            raise RuntimeError("Client is not initialized. Please call init_client() first.")
        devices = []
        device_data = []
        all_device_data = []
        if channel_idx and not isinstance(channel_idx, list):
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
                if device.device_type == DeviceType.Emg or device.device_type == DeviceType.DelsysGogniometer:
                    queue_name = [key for key in self.sdk_client.all_queue if "emg" in key][0]
                elif device.device_type == DeviceType.Imu or device.device_type == DeviceType.DelsysGogniometer:
                    queue_name = [key for key in self.sdk_client.all_queue if "aux" in key][0]
                device.new_data, _ = self.sdk_client.all_queue[queue_name].get()

            if channel_idx:
                device_data = np.ndarray((len(channel_idx), device.new_data.shape[1]))
                for i, idx in enumerate(channel_idx):
                    device_data[i, :] = device.new_data[idx, :]
            device_data = device_data if channel_idx else device.new_data
            if get_frame:
                device.append_data(device.new_data)
            all_device_data.append(device_data)
            if len(all_device_data) == 1:
                all_device_data = all_device_data[0]
        return all_device_data

    def get_frame(self):
        """
        Get a frame from the interface. This function is used to get data from the interface.
        """
        if not self.is_initialized:
            self.init_client()
            # raise RuntimeError("Client is not initialized. Please call init_client() first.")
        self.get_device_data(get_frame=True)
        return True

    def check_rate_devices(self):
        for device in self.devices:
            if "emg" in device.device_type.value:
                device.rate = self.sdk_client.get_emg_streaming_rate()
            if "imu" in device.device_type.value:
                device.rate = self.sdk_client.get_aux_streaming_rate()

    def init_client(self):
        """
        Initialize the client if it's not already done. This function has to be called before getting a frame.
        """
        self.is_initialized = True
        device_types = np.unique(np.array([device.device_type.value for device in self.devices]))
        self.sdk_client.initialize_sensors(device_types)
        self.check_rate_devices()
        if self.sdk_client._threads_to_run["avanti_emg"] and self.sdk_client._threads_to_run["legacy_emg"]:
            raise (
                RuntimeError(
                    "Both avanti and legacy emg type are paired. "
                    "It is not possible to use both at the same time in biosiglive for now."
                )
            )
        self.sdk_client.start_streaming()
