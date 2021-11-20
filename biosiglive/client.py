import socket, pickle
import json
import struct

Buff_size = 4096


class Message:
    def __init__(self):
        self.command = []
        self.dic = {}
        self.dic["command"] = []
        self.dic["nb_frame_of_interest"] = 7
        self.dic["read_frequency"] = 33  # frequency at which the ocp should run
        self.dic["emg_windows"] = 2000
        self.dic["get_names"] = False
        self.dic["nb_of_data_to_export"] = None
        # self.dic["EMG_unit"] = "V"  # or "mV"


class Client:
    def __init__(self, server_address, port, type="TCP", name=None):
        self.name = name if name is not None else "Client"
        self.type = type
        self.address = f"{server_address}:{port}"
        if self.type == "TCP":
            self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        elif self.type == "UDP":
            self.client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        else:
            raise RuntimeError(f"Invalid type of connexion ({ self.type}). Type must be 'TCP' or 'UDP'.")
        self.client.connect((server_address, port))

    def connect(self, server_address, port):
        self.client.connect((server_address, port))

    @staticmethod
    def client_sock(type):
        if type == "TCP" or type is None:
            return socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        elif type == "UDP":
            return socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # TODO: add possibility to ask for some index
    def recv_all(self, buff_size):
        msg_len = self.client.recv(4)
        msg_len = struct.unpack('>I', msg_len)[0]
        data = []
        l = 0
        while l < msg_len:
            chunk = self.client.recv(buff_size)
            l += len(chunk)
            data.append(chunk)
        data = b"".join(data)
        data = json.loads(data)
        return data

    def get_data(
        self,
        data,
        nb_frame_of_interest=7,
        read_frequency=33,
        emg_wind=2000,
        nb_of_data_to_export=None,
        buff=Buff_size,
        get_kalman=False,
        get_names=False,
        mvc_list=None,
        norm_emg=None,
        raw=False,
    ):

        message = Message()
        message.dic["get_names"] = get_names
        message.dic["norm_emg"] = norm_emg
        message.dic["mvc_list"] = mvc_list
        message.dic["kalman"] = get_kalman
        message.dic["nb_frame_of_interest"] = nb_frame_of_interest
        message.dic["read_frequency"] = read_frequency
        message.dic["emg_windows"] = emg_wind
        message.dic["nb_of_data_to_export"] = nb_of_data_to_export
        message.dic["raw_data"] = raw

        if norm_emg is True and mvc_list is None:
            raise RuntimeError("Define a list of MVC to normalize the EMG data.")
        elif mvc_list is not None and norm_emg is not True:
            print(
                "[WARNING] You have defined a list of MVC but not asked for normalization. "
                "Please turn norm_EMG to True tu normalize your data."
            )

        message.dic["command"] = []
        for i in data:
            message.dic["command"].append(i)
            if i != "emg" and i != "markers" and i != "imu" and i != "force_plate":
                raise RuntimeError(f"Unknown command '{i}'. Command must be :'emg', 'markers' or 'imu' ")

        Client.send(self, json.dumps(message.dic).encode(), type="all")
        return self.recv_all(buff)

    def send(self, data, type=None, IP=None, port=None):
        data = pickle.dumps(data) if not isinstance(data, bytes) else data
        if type == None:
            return self.client.send(data)
        elif type == "all":
            return self.client.sendall(data)
        elif type == "to":
            return self.client.sendto(data, address=(IP, port))

    def close(self):
        return self.client.close


if __name__ == "__main__":
    from time import time, sleep

    host_ip = "192.168.1.211"
    # host_ip = 'localhost'
    host_port = 50000

    tic = time()
    client = Client(host_ip, host_port, "TCP")
    data = client.get_data(
        data=["markers"],
        nb_frame_of_interest=7,
        read_frequency=33,
        emg_wind=2000,
        nb_of_data_to_export=8,
        get_names=True,
        get_kalman=True,
    )
    print(data["kalman"])
    print(time() - tic)
