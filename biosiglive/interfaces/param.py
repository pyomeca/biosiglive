from math import ceil

class Type:
    def __init__(self, name: str = None, type: str = None, rate: float = None, system_rate: float = 100):
        self.name = name
        self.type = type
        self.rate = rate
        self.system_rate = system_rate
        self.sample = ceil(rate / self.system_rate)
        self.range = None

    def set_name(self, name: str):
        self.name = name

    def set_type(self, type: str):
        self.type = type

    def set_rate(self, rate: int):
        self.rate = rate


class Device(Type):
    """
    This class is used to store the available devices.
    """

    def __init__(self, name: str = None, type: str = "emg", rate: float = None):
        super().__init__(name, type, rate)
        self.infos = None


class Imu(Type):
    """
    This class is used to store the available IMU devices.
    """
    def __init__(self, name: str = None, rate: float = None, from_emg: bool = False):
        type = "imu" if not from_emg else "imu_from_emg"
        super().__init__(name, type, rate)


class MarkerSet(Type):
    """
    This class is used to store the available markers.
    """
    def __init__(self, name: str = None, rate: float = None, unlabeled: bool = False):
        type = "unlabeled" if unlabeled else "labeled"
        super().__init__(name, type, rate)
        self.markers_names = name
        self.subject_name = None



if __name__ == '__main__':
    device = Imu("test", 145.1, True)
    print(device.sample)