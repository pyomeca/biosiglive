from enum import Enum
import numpy as np


class Type(Enum):
    Avanti = "O"
    Legacy = "A"
    AvantiGogniometer = "23"


class Sensor:
    def __init__(self, index, trigno_box=None, buff_size=100):
        self.buff_size = buff_size
        self.name = f"sensor {index}"
        self.index = index
        self.type = None
        self.mode = None
        self.units = None
        self.type = None
        self.range = None
        self.nb_emg_channels = None
        self.nb_aux_channels = None
        self.max_emg_samples = None
        self.max_aux_samples = None
        self.emg_rate = None
        self.aux_rate = None
        self._index_emg = 0
        self._index_aux = 0
        self._emg_frame_numbers = []
        self._aux_frame_numbers = []
        self.sensor_start_idx = 0

        self.emg_buffer = None
        self.aux_buffer = None
        self.trigno_box = trigno_box

        if trigno_box is not None:
            self.initialize()

    def emg_aux_type(self):
        if self.is_paired:
            emg = self.type if self.nb_emg_channels > 0 else None
            aux = self.type if self.nb_aux_channels > 0 else None
            return emg, aux
        else:
            return None, None

    def initialize(self):
        """
        Initialize the sensor
        :param trigno_box: TrignoBox object
        :return: None
        """
        if not self.is_sensor_paired():
            self.is_paired = False
            return

        self.is_paired = True
        self.mode = self.get_sensor_mode()
        # self.units = self.get_sensor_units()
        self.type = self.get_sensor_type()
        self.nb_emg_channels = self.get_sensor_emgchannel()
        self.nb_aux_channels = self.get_sensor_auxchannel()
        self.aux_rate = self.get_aux_streaming_rate()
        self.emg_rate = self.get_emg_streaming_rate()
        self.sensor_start_idx = self.get_sensor_idx()
        self.emg_range = (self.sensor_start_idx, self.sensor_start_idx + self.nb_emg_channels)
        self.aux_range = (self.sensor_start_idx * 9, self.sensor_start_idx * 9 + self.nb_aux_channels)

        # self.emg_buffer = np.empty((self.nb_emg_channels, self.max_emg_samples, self.buff_size))
        # self.aux_buffer = np.empty((self.nb_aux_channels, self.max_aux_samples, self.buff_size))

    @property
    def last_emg_chunck(self):
        return self.emg_buffer[..., self._index_emg]

    @property
    def last_aux_chunck(self):
        return self.aux_buffer[..., self._index_aux]

    def update_emg_buffer(self, emg_data, n_chunck=None):
        if not self.is_paired:
            return
        self.extend_frame_numbers(self._emg_frame_numbers, n_chunck, self.max_emg_samples)
        self.emg_buffer[..., self._index_emg] = emg_data
        self._index_emg = (self._index_emg + 1) % self.buff_size

    def get_emg_streaming_rate(self):
        if int(self.trigno_box.send_command(f"SENSOR {self.index} EMGCHANNELCOUNT?")) > 0:
            return float(self.trigno_box.send_command(f"SENSOR {self.index} CHANNEL 1 RATE?"))
        else:
            return None

    def get_aux_streaming_rate(self):
        if int(self.trigno_box.send_command(f"SENSOR {self.index} AUXCHANNELCOUNT?")) > 0:
            return float(self.trigno_box.send_command(f"SENSOR {self.index} CHANNEL 2 RATE?"))
        else:
            return None

    def update_aux_buffer(self, aux_data, n_chunck=None):
        if not self.is_paired:
            return
        self.extend_frame_numbers(self._aux_frame_numbers, n_chunck, self.max_aux_samples)
        self.aux_buffer[..., self._index_aux] = aux_data
        self._index_aux = (self._index_aux + 1) % self.buff_size

    @property
    def emg_frame_numbers(self):
        return self._emg_frame_numbers

    @property
    def aux_frame_numbers(self):
        return self._aux_frame_numbers

    def extend_frame_numbers(self, init_list, n_chunck, n_samples):
        if n_chunck is None:
            init_list.extend(list(range(len(init_list), len(init_list) + n_samples)))
        else:
            init_list.extend(list(range(n_chunck, n_chunck + n_samples)))
        return init_list

    def get_emg_from_buffer(self):
        return np.roll(self.emg_buffer, -self._index_emg, axis=-1).reshape((self.nb_emg_channels, -1))

    def get_aux_from_buffer(self):
        return np.roll(self.aux_buffer, -self._index_aux, axis=-1).reshape((self.nb_aux_channels, -1))

    def get_sensor_info(self, info="TYPE"):
        return self.trigno_box.send_command(f"SENSOR {self.index} {info}?")

    def get_sensor_type(self):
        sensor_type = self.trigno_box.send_command(f"SENSOR {self.index} TYPE?")
        try:
            return Type(sensor_type)
        except ValueError:
            raise ValueError(f"Sensor {self.index} has an unknown type: {sensor_type}")

    def get_sensor_mode(self):
        return self.trigno_box.send_command(f"SENSOR {self.index} MODE?")

    def get_sensor_emgchannel(self):
        if not self.is_sensor_paired():
            return 0
        return int(self.trigno_box.send_command(f"SENSOR {self.index} EMGCHANNELCOUNT?"))

    def get_sensor_auxchannel(self):
        if not self.is_sensor_paired():
            return 0
        return int(self.trigno_box.send_command(f"SENSOR {self.index} AUXCHANNELCOUNT?"))

    def is_sensor_paired(self):
        return self.trigno_box.send_command(f"SENSOR {self.index} PAIRED?") == "YES"

    def get_sensor_idx(self):
        return int(self.trigno_box.send_command(f"SENSOR {self.index} STARTINDEX?"))
