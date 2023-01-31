"""
Make sure before using this example that a server is running, if you don't have one, please run the "server.py" example.
This example shows how to get data from a biosiglive server using the biosiglive client.
Here the client connection is used as an interface of the GenericInterface class.
To initialize the interface, you must give the server IP address (IP and port) and the target frequency at which the data will be broadcast. Please use "127.0.0.1" instead of "localhost" if you are using the server on the same computer as the client, to avoid connection problems.
Once the interface is initialized, you can add devices to it.
The device arguments are:
- nb_channels: int
         Number of device channels.
- command_name: str
         Name of the command to send to the server to get the data. The command must be the same as the one used in the data dictionary given to the server.
- name: str
         Device name.
- rate: float
         Device frequency.

A set of markers can also be added to the interface. The arguments of the marker set are as follows
- number of markers: int
         Number of markers in the marker set.
     name: str
         Marker set name.
- rate: float
         Marker set frequency.
- command_name: str
         Name of the command to send to the server to get the data. The command must be the same as the one used in the data dictionary given to the server.

Once the devices and marker sets have been added to the interface, you can start the interface.
After initializing the interface, the data exchange is done in a loop.
What is important here is that data can be retrieved from the server in two ways:
- using the get_data_from_server() method of the interface. In this method, you specify the command to send to the server to get the data. The command must be the same as the one used in the data dictionary given to the server. You can specify the number of frames to get from the server (last, last 10, etc., or all frames). You can also specify a downsampling factor to reduce the number of frames. The last two arguments aim to reduce the amount of data to get from the server. This method is used to have all server data at the same time to synchronize data.
- by using the get-device_data or get_marker_set_data methods of the interface. In this case, you do not need to specify the command because it has already been defined when adding the device or marker to the interface. In this case, you can either get the data directly from the server using the get from server argument, or get the data from the last call of the get_data_from_server() method. If you want synchronized data, you must call the get_data_from_server() method before retrieving data from the device or marker set. Please note that if you have read data from the server once, you cannot read it again. If you want to read data through multiple clients, you must use the same number of servers.
"""

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
            command=["proc_device_data", "marker_set_data"], nb_frame_to_get=10, down_sampling={"proc_device_data": 5}
        )
        data = tcp_client.get_device_data(device_name="processed EMG", get_from_server=False)
        data_mark = tcp_client.get_marker_set_data(marker_set_name="markers", get_from_server=False)
        time.sleep(0.01)
        print(data)
        print(data_mark)
