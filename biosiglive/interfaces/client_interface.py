import numpy as np
from .param import Device, MarkerSet
from ..streaming.client import Client
from typing import Union
try:
    from vicon_dssdk import ViconDataStream as VDS
except ModuleNotFoundError:
    pass


class TcpClient(Client):
    """
    Class for interfacing with the Vicon system.
    """
    def __init__(self, ip: str = None, port: int = 801, type: str = "TCP", read_frequency: int = 100):
        ip = ip if ip else "127.0.0.1"
        self.devices = []
        self.imu = []
        self.markers = []
        self.data_to_stream = []
        self.read_frequency = read_frequency
        super().__init__(ip, port, type)
        self.init_command(data=[], read_frequency=self.read_frequency, emg_wind=2000, nb_frames_to_get=1,
                                 get_kalman=False, get_names=False, mvc_list=None, norm_emg=False, raw=True)

    def add_device(self, name: str, type: str = "emg", rate: float = 2000):
        device_tmp = Device(name, type, rate)
        self.devices.append(device_tmp)

    # def add_imu(self, name: str, rate: int = 148.1, from_emg: bool = False):
    #     self.imu.append(Imu(name, rate, from_emg=from_emg))

    def add_markers(self, name: str = None, rate: int = 100, unlabeled: bool = False, subject_name: str = None):
        markers_tmp = MarkerSet(name, rate, unlabeled)
        markers_tmp.subject_name = subject_name
        markers_tmp.markers_names = name
        self.markers.append(markers_tmp)

    def get_data_from_server(self):
        all_data = []
        if len(self.data_to_stream) == 0:
            raise ValueError("No data to stream")
        self.client.update_command(name="command", value=self.data_to_stream)
        data = self.client.get_data()
        for stream_data in self.data_to_stream:
            for key in data:
                if key == stream_data:
                    all_data.append(np.array(data[key]))
        return all_data

    def get_device_data(self, device_name: str = "all", stream_now: bool = False, get_names: bool = False, *args):
        devices = []
        all_device_data = []
        if not isinstance(device_name, list):
            device_name = [device_name]

        if device_name != "all":
            for d, device in enumerate(self.devices):
                if device.name == device_name[d]:
                    devices.append(device)
        else:
            devices = self.devices

        if stream_now:
            self.client.update_command(name="get_names", value=get_names)
            self.client.update_command(name="command", value=[i.type for i in devices])
            data = self.client.get_data()
            for device in devices:
                for key in data:
                    if key == device.type:
                        all_device_data.append(np.array(data[key]))
            return all_device_data

        else:
            for device in devices:
                self.data_to_stream.append(device.type)
            return None

    def get_markers_data(self, stream_now: bool = False, get_names: bool = False):
        if stream_now:
            self.client.update_command(name="get_names", value=get_names)
            self.client.update_command(name="command", value="markers")
            data = self.client.get_data()
            return np.array(data["markers"])
        else:
            self.data_to_stream.append("markers")
            return None



