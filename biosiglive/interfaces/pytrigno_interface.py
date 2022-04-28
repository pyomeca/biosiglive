from .param import *
import pytrigno


class PytrignoClient:
    def __init__(self, ip: str = None):
        self.address = "127.0.0.1" if not ip else ip
        self.devices = []
        self.imu = []

    def _add_device(self, name: str = None, range: tuple = (0, 16), type: str = "emg", rate: float = 2000):
        new_device = Device(name, type, rate)
        device.range = range
        self.devices.append(new_device)
        if device.type == "emg":
            self.emg_client = pytrigno.TrignoEMG(
                channel_range=device.range, samples_per_read=device.sample, host=self.address
            )
            self.emg_client.start()

        elif device.type == "imu":
            imu_range = (device.range[0] * 9, device.range[1] * 9)
            self.imu_client = pytrigno.TrignoIM(
                channel_range=imu_range, samples_per_read=device.sample, host=self.address
            )
            self.imu_client.start()

        else:
            raise RuntimeError("Device type must be 'emg' or 'imu' with pytrigno.")

    def get_device_data(self, device_name: str = None):
        devices = []
        all_device_data = []

        if device_name and not isinstance(device_name, list):
            device_name = [device_name]

        if device_name:
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


if __name__ == '__main__':
    client = PytrignoClient("")
    client._add_device(range=(0, 16), type="emg", rate=2000)
    client._add_device(range=(0, 16), type="imu", rate=148.1)

