from biosiglive import (
    TcpClient,
    DeviceType,
)
import time

if __name__ == "__main__":
    server_ip = "127.0.0.1"
    server_port = 50000
    tcp_client = TcpClient(server_ip, server_port, read_frequency=100)
    tcp_client.add_device(
        5, command_name="proc_device_data", device_type=DeviceType.Emg, name="processed EMG", rate=2000
    )
    tcp_client.add_marker_set(15, name="markers", rate=100, command_name="marker_set_data")
    while True:
        tcp_client.get_data_from_server(
            command=["proc_device_data", "marker_set_data"],
            nb_frame_to_get=1,
        )
        data = tcp_client.get_device_data(device_name="processed EMG", get_from_server=False)
        data_mark = tcp_client.get_marker_set_data(marker_set_name="markers", get_from_server=False)
        time.sleep(0.01)
        print(data)
        print(data_mark)
