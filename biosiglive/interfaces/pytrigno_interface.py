from .param import *
try:
    import pytrigno
except ModuleNotFoundError:
    pass


class PytrignoClient:
    def __init__(self, ip: str = None):
        self.address = "127.0.0.1" if not ip else ip
        self.devices = []
        self.imu = []
        self.markers = []

        self.emg_client, self.imu_client = None, None
        self.is_frame = False

    def add_device(self, name: str = None, range: tuple = (0, 16), type: str = "emg", rate: float = 2000, real_time: bool = False):
        """
        Add a device to the Pytrigno client.
        Parameters
        ----------
        name : str
            Name of the device.
        range : tuple
            Range of the electrodes.
        type : str
            Type of the device. (emg or imu)
        rate : float
            Rate of the device.
        real_time : bool
            If true device  will be used in real time application
        """
        new_device = Device(name, type, rate)
        new_device.range = range
        self.devices.append(new_device)
        if type == "emg":
            self.emg_client = pytrigno.TrignoEMG(
                channel_range=new_device.range, samples_per_read=new_device.sample, host=self.address
            )
            self.emg_client.start()

        elif new_device.type == "imu":
            imu_range = (new_device.range[0] * 9, new_device.range[1] * 9)
            self.imu_client = pytrigno.TrignoIM(
                channel_range=imu_range, samples_per_read=new_device.sample, host=self.address
            )
            self.imu_client.start()

        else:
            raise RuntimeError("Device type must be 'emg' or 'imu' with pytrigno.")

    def get_device_data(self, device_name: str = "all", *args):
        """
        Get data from the device.
        Parameters
        ----------
        device_name : str
            Name of the device.
        *args
            Additional argument.

        Returns
        -------
        data : list
            Data from the device.
        """
        devices = []
        all_device_data = []

        if device_name and not isinstance(device_name, list):
            device_name = [device_name]

        if device_name != "all":
            for d, device in enumerate(self.devices):
                if device.name and device.name == device_name[d]:
                    devices.append(device)
        else:
            devices = self.devices

        for device in devices:
            if device.type == "emg":
                device_data = self.emg_client.read()
            elif device.type == "imu":
                device_data = self.imu_client.read()
            else:
                raise RuntimeError(f"Device type ({device.type}) not supported with pytrigno.")
            all_device_data.append(device_data)
        return all_device_data

    def get_markers_data(self, marker_names: list = None, subject_name: str = None):
        raise RuntimeError("It's not possible to get markers data from pytrigno.")

    def get_force_plate_data(self):
        raise RuntimeError("It's not possible to get force plate data from pytrigno.")

    def add_markers(self, name: str = None, rate: float = 100, unlabeled: bool = False, subject_name: str = None):
        raise RuntimeError("It's not possible to get markers data from pytrigno.")


    @staticmethod
    def init_client():
        pass

    @staticmethod
    def get_latency():
        return 0

    @staticmethod
    def get_frame():
        return True

