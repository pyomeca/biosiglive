import socket
import struct
import threading
from queue import Queue
import time

import numpy as np

from .enums import AvantiSensor, LegacySensor, Enum
from .sensor import Sensor, Type


BYTES_PER_CHANNEL = 4
CMD_TERM = "\r\n\r\n"
EMG_SAMPLE_RATE = 2000
AUX_SAMPLE_RATE = 148.148
SAMPLE_INTERVAL = 0.0135


class TrignoSDKClient:
    def __init__(
        self,
        host="127.0.0.1",
        cmd_port=50040,
        timeout=2.0,
        fast_mode=False,
        buffer_size=1000,
        stream_rate=74.074074,
        init_sensors=True,
    ):
        self.buffer_size = buffer_size
        self.targeted_rate = stream_rate
        self.host = host
        self.cmd_port = cmd_port
        self.timeout = timeout
        self.is_connected = False
        self._comm_socket = None
        self.fast_mode = fast_mode
        self.avanti_emg_socket = None
        self.avanti_aux_socket = None
        self.legacy_emg_socket = None
        self.legacy_aux_socket = None
        self.time_counter = time.perf_counter
        self.all_socket = {}
        self.all_events = {}
        self._last_data = {
            "avanti_emg": None,
            "avanti_aux": None,
            "legacy_emg": None,
            "legacy_aux": None,
        }

        self.avanti_emg_queue = Queue()
        self.avanti_aux_queue = Queue()
        self.legacy_emg_queue = Queue()
        self.legacy_aux_queue = Queue()
        self.all_queue = {
            "avanti_emg": self.avanti_emg_queue,
            "avanti_aux": self.avanti_aux_queue,
            "legacy_emg": self.legacy_emg_queue,
            "legacy_aux": self.legacy_aux_queue,
        }

        if self.fast_mode:
            print("Warning: Fast mode enabled. Responses will not be waited for.")

        self.connect(init_sensors=init_sensors)

    def connect(self, init_sensors=True):
        """Establish connection to Trigno SDK command port."""
        self._comm_socket = socket.create_connection((self.host, self.cmd_port), self.timeout)
        self.is_connected = True
        self.send_command("BACKWARDS COMPATIBILITY OFF")
        if init_sensors:
            self.initialize_sensors()

        try:
            _ = self._comm_socket.recv(1024)
        except socket.timeout:
            pass

    def reconnect_device(self):
        for _, item in self.all_socket.items():
            if item is not None:
                item.close()
        for _, event in self.all_events.items():
            if event is not None:
                event = None
        self.initialize_sensors()

    def initiate_data_connection(self):
        for key, item in self._threads_to_run.items():
            self.all_events[key] = item if not item else threading.Event()
            self.all_socket[key] = item if not item else self._connect_to_socket(self._port_from_name(key))

    def _port_from_name(self, name):
        if "emg" in name:
            return AvantiSensor().emg_port if "avanti" in name.lower() else LegacySensor().emg_port
        elif "aux" in name:
            return AvantiSensor().aux_port if "avanti" in name.lower() else LegacySensor().aux_port
        else:
            raise ValueError("Type not recognized.")

    def _check_data_to_stream(self, data_to_stream):
        if "emg" not in data_to_stream and "delsys_gogniometer" not in data_to_stream:
            self._threads_to_run["avanti_emg"] = False
            self._threads_to_run["legacy_emg"] = False
        if "aux" not in data_to_stream and "delsys_gogniometer" not in data_to_stream:
            self._threads_to_run["avanti_aux"] = False
            self._threads_to_run["legacy_aux"] = False

    def initialize_sensors(self, data_to_stream=None):
        """Initialize all sensors."""
        self.sensors = [Sensor(i, self, self.buffer_size) for i in range(1, 17)]
        self._get_which_thread_to_run()
        if data_to_stream is not None:
            self._check_data_to_stream(data_to_stream)
        self.initiate_data_connection()

    def _get_which_thread_to_run(self):
        all_emg = []
        all_aux = []
        for sensor in self.sensors:
            if not sensor.is_paired:
                continue
            emg, aux = sensor.emg_aux_type()
            all_emg.append(emg)
            all_aux.append(aux)

        self._threads_to_run = {
            "avanti_emg": True in ["Avanti" in _.name for _ in all_emg if _ is not None],
            "avanti_aux": True in ["Avanti" in _.name for _ in all_aux if _ is not None],
            "legacy_emg": True in ["Legacy" in _.name for _ in all_emg if _ is not None],
            "legacy_aux": True in ["Legacy" in _.name for _ in all_aux if _ is not None],
        }

    def _connect_to_socket(self, port):
        _data_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        _data_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        _data_socket.connect((self.host, port))
        _data_socket.setblocking(False)
        return _data_socket

    def _bytes_per_sample(self, n_channels):
        """Return the number of bytes per sample for a given number of channels."""
        return n_channels * BYTES_PER_CHANNEL

    def buffer_size(self, n_channels, n_samples):
        """Return the size of the buffer required to store a given number of samples for a given number of channels."""
        return self._bytes_per_sample(n_channels) * n_samples

    def disconnect(self):
        """Close the socket connection."""
        if self._comm_socket:
            self._comm_socket.close()
            self._comm_socket = None

    def send_command(self, command: str) -> str:
        """
        Send a command or query to the Trigno system and return the response as a string.
        Command strings must already include any needed arguments.
        """
        if self._comm_socket is None:
            raise RuntimeError("Not connected. Call connect() first.")

        full_command = f"{command}\r\n\r\n"
        self._comm_socket.sendall(full_command.encode("ascii"))

        # If in fast mode, return immediately without waiting for a response
        if self.fast_mode:
            return ""

        # Give the server time to respond
        while True:
            time.sleep(0.1)
            try:
                response = self._comm_socket.recv(1024)
                return response.decode("ascii").strip()
            except socket.timeout:
                None

    def read(self, connection, buffer_size, n_channels, sample_interval, data_matrix=None):
        l = -1
        packet = bytes()
        flush_packet = bytes()
        time_counter = self.time_counter()
        try:
            while l != 0:
                flush_packet = connection.recv(4096)
                packet += flush_packet
                l = len(flush_packet)
        except BlockingIOError:
            pass

        if data_matrix is None:
            data_matrix = np.zeros((int(buffer_size // 4 / n_channels), n_channels))
        data = np.asarray(struct.unpack("<" + "f" * (len(packet) // 4), packet))
        nb_flushed_data = int(buffer_size // 4 / n_channels) - len(data) // n_channels
        if nb_flushed_data > 0:
            time_counter += nb_flushed_data * sample_interval
        data_matrix[: len(data) // n_channels, :] = data.reshape((-1, n_channels))[
            -int(buffer_size // 4 / n_channels) :, :
        ]

        return data_matrix.T, time_counter

    def start_streaming(self):
        # self.streaming_rate = self._compute_rate_from_sensors(rate)
        start_response = self.send_command("START")
        is_started = start_response == "OK"
        if not is_started and not self.fast_mode:
            raise RuntimeError("Streaming not started.")
        self._launch_threads()

    def _compute_rate_from_sensors(self, initial_rate):
        # get minimum max samples
        closest_rate = initial_rate
        sample_numbers = min(self.get_max_aux_samples(), self.get_max_emg_samples)
        n_sample_init = sample_numbers * (1 / initial_rate) / SAMPLE_INTERVAL
        if n_sample_init % 1 != 0:
            closest_rate = 1 / (int(n_sample_init) * SAMPLE_INTERVAL / sample_numbers)
            print(
                f"WARNING: You requested an unvalid sampling rate ({initial_rate} Hz)."
                f"The rate was set to the closest valid rate: {closest_rate:.3f} Hz."
            )

        return closest_rate

    def buffer_size_for_type(self, name, n_samples=None):
        if "emg" in name:
            n_samples = self.get_max_emg_samples() if not n_samples else n_samples
            n_channel = 16
            buffer_size = n_channel * n_samples * BYTES_PER_CHANNEL
        elif "aux" in name:
            n_samples = self.get_max_aux_samples() if not n_samples else n_samples
            n_channel = 144 if "avanti" in name.lower() else 48
            buffer_size = n_channel * n_samples * BYTES_PER_CHANNEL
        else:
            raise RuntimeError("Invalid sensor type.")
        return buffer_size, n_channel, n_samples

    @staticmethod
    def flush_socket(socket):
        try:
            while True:
                socket.recv(4096)
        except BlockingIOError:
            return

    def _launch_one_thread(self, socket_tmp, name, data_queue, event):
        buffer_size, n_channels, n_samples = self.buffer_size_for_type(name)
        data_matrix = np.zeros((int(buffer_size // 4 / n_channels), n_channels))
        channel_native_streaming_rate = (
            self.get_emg_streaming_rate() if "emg" in name else self.get_aux_streaming_rate()
        )
        sample_interval = 1 / channel_native_streaming_rate
        streaming_interval = self._get_interval_from_channel(name)
        self.flush_socket(socket_tmp)

        def _read_function():
            data, timestamp = self.read(socket_tmp, buffer_size, n_channels, sample_interval, data_matrix)
            try:
                data_queue.get_nowait()
            except:
                pass
            data_queue.put_nowait((data[np.unique(data.nonzero()[0])], timestamp))

        def _thread_func():
            while True:
                # tic = self.time_counter()
                _read_function()
                # toc = self.time_counter()
                # if toc - tic < streaming_interval:
                #     time.sleep(streaming_interval - (toc - tic))
                # else:
                #     print(f"Warning: Streaming interval exceeded. Time taken to read data: {toc - tic:.3f} s, target time: {streaming_interval:.3f} s.")

        thread = threading.Thread(target=_thread_func, name=name)
        thread.start()

    def _get_interval_from_channel(self, name):
        max_samples = self.get_max_emg_samples() if "emg" in name else self.get_max_aux_samples()
        sample = int(max_samples * (1 / self.targeted_rate) / SAMPLE_INTERVAL)
        return sample * SAMPLE_INTERVAL / max_samples

    def _launch_threads(self):
        all_soc, all_q, all_ev = self.all_socket, self.all_queue, self.all_events
        self._all_threads = [
            self._launch_one_thread(all_soc[n], n, all_q[n], all_ev[n])
            for n in self._threads_to_run.keys()
            if self._threads_to_run[n]
        ]

        # def _main_thread_func():
        #     while True:
        #         for name in self.all_socket.keys():
        #             self.all_events[name].wait()
        #             self.all_events[name].clear()
        #         self._set_all_data()

        # main_thread = threading.Thread(target=_main_thread_func, name='main')
        # main_thread.start()

    def stop_streaming(self):
        return self.send_command("STOP")

    def disconnect(self):
        self.stop_streaming()
        _ = [socket.close() for socket in self.all_socket.values()]
        self._comm_socket.close()

    def get_emg_streaming_rate(self):
        return float(self.send_command("MAX SAMPLES EMG")) / 0.0135

    def get_aux_streaming_rate(self):
        return float(self.send_command("MAX SAMPLES AUX")) / 0.0135

    def get_max_emg_samples(self):
        return int(self.send_command("MAX SAMPLES EMG"))

    def get_max_aux_samples(self):
        return int(self.send_command("MAX SAMPLES AUX"))

    def get_aux_streaming_rate(self):
        return int(self.send_command("MAX SAMPLES AUX")) / 0.0135

    def get_trigger_state(self):
        return self.send_command("TRIGGER?")

    def set_trigger(self, which="START", state="ON"):
        return self.send_command(f"TRIGGER {which} {state}")

    def get_backwards_compatibility(self):
        return self.send_command("BACKWARDS COMPATIBILITY?")

    def set_backwards_compatibility(self, state="ON"):
        return self.send_command(f"BACKWARDS COMPATIBILITY {state}")

    def get_upsampling(self):
        return self.send_command("UPSAMPLING?")

    def set_upsampling(self, state="ON"):
        return self.send_command(f"UPSAMPLE {state}")

    def get_sensor_info(self, n, info="TYPE"):
        return self.send_command(f"SENSOR {n} {info}?")

    def get_sensor_emgchannel(self, n):
        if not self.is_sensor_paired(n):
            return 0
        return int(self.send_command(f"SENSOR {n} EMGCHANNELCOUNT?"))

    def get_sensor_auxchannel(self, n):
        if not self.is_sensor_paired(n):
            return 0
        return int(self.send_command(f"SENSOR {n} AUXCHANNELCOUNT?"))

    def is_sensor_paired(self, n):
        return self.send_command(f"SENSOR {n} PAIRED?") == "YES"

    def get_sensor_idx(self, n):
        return int(self.send_command(f"SENSOR {n} STARTINDEX?"))

    def get_list_sensors_and_idx(self):
        sensors = []
        for i in range(1, 16):
            if self.is_sensor_paired(i):
                sensors.append([self.get_sensor_info(i, "TYPE"), self.get_sensor_idx(i)])
            else:
                sensors.append([None, None])
        return sensors

    def pair_sensor(self, n):
        return self.send_command(f"SENSOR {n} PAIR")

    def set_sensor_mode(self, n, mode):
        return self.send_command(f"SENSOR {n} SETMODE {mode}")

    def get_endianness(self):
        return self.send_command("ENDIANNESS?")

    def set_endianness(self, mode="LITTLE"):
        return self.send_command(f"ENDIAN {mode}")

    def get_base_serial(self):
        return self.send_command("BASE SERIAL?")

    def get_base_firmware(self):
        return self.send_command("BASE FIRMWARE?")

    def get_number_emgchannel(self):
        nb_channel = 0
        for i in range(1, 16):
            nb_channel += self.get_sensor_emgchannel(i)
        return nb_channel

    def get_number_auxchannel(self):
        nb_channel = 0
        for i in range(1, 16):
            nb_channel += self.get_sensor_auxchannel(i)
        return nb_channel

    def _set_avanti_emg_data(self):
        try:
            data = self.avanti_emg_queue.get_nowait()
            for sensor in self.sensors:
                if sensor.type == Type.Avanti or sensor.type == Type.AvantiGogniometer:
                    sensor.update_emg_buffer(data[0][sensor.emg_range[0] : sensor.emg_range[1], :], data[1])
        except:
            return

    def _set_avanti_aux_data(self):
        try:
            data = self.avanti_aux_queue.get_nowait()
            for sensor in self.sensors:
                if sensor.type == Type.Avanti or sensor.type == Type.AvantiGogniometer:
                    sensor.update_aux_buffer(data[0][sensor.aux_range[0] : sensor.aux_range[1], :], data[1])
        except:
            return

    def _set_legacy_emg_data(self):
        try:
            data = self.legacy_emg_queue.get_nowait()
            for sensor in self.sensors:
                if sensor.type == Type.Legacy:
                    sensor.update_emg_buffer(data[0][sensor.emg_range[0] : sensor.emg_range[1], :], data[1])
        except:
            return

    def _set_legacy_aux_data(self):
        try:
            data = self.legacy_aux_queue.get_nowait()
            for sensor in self.sensors:
                if sensor.type == Type.Legacy:
                    sensor.update_aux_buffer(data[0][sensor.aux_range[0] : sensor.aux_range[1], :], data[1])
        except:
            return

    def _set_all_data(self):
        self._set_avanti_emg_data(),
        self._set_avanti_aux_data(),
        self._set_legacy_emg_data(),
        self._set_legacy_aux_data(),
