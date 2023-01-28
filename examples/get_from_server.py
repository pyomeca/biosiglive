"""
Make sure before using this example a server is running, if you don't have one please run the example "server.py".
This example shows how to get data from a biosiglive server using the biosiglive client.
Here the is used as an interface from the GenericInterface class.
To initialize the interface you have to give the ip address of the server (IP and port) and the target frequency
 at which the data will be streamed. Please use the address "127.0.0.1" instead of "localhost" if you are using
 the server on the same computer as the client. to avoid any problem with the connection.
Once the interface is initialized you can add devices to it.
The device arguments are:
    - nb_channels: int
        Number of channels of the device.
    - command_name: str
        Name of the command to send to the server to get the data. The command needs to be the same as used in the
        data dictionary gives to the server.
    - name: str
        Name of the device.
    - rate: float
        Frequency of the device.
A marker set can also be added to the interface as well. The marker set arguments are:
    - number of markers: int
        Number of markers of the marker set.
    name: str
        Name of the marker set.
    - rate: float
        Frequency of the marker set.
    command_name: str
        Name of the command to send to the server to get the data. The command needs to be the same as used in the data
        dictionary gives to the server.
Once the devices and marker sets are added to the interface you can start the interface.
After initializing the interface the data streaming take place in a loop.
What's important here is that data can be retrieved from the server in two ways:
    - using the get_data_from_server() method of the interface. In this method you specify the command to send to the
    server to get the data. The command needs to be the same as used in the data dictionary gives to the server.
     You can specify the number of frame to get from the server (the last one, the 10 last, etc., or alls the frames).
     Also you can psecify a downsampling factor to reduce the number of frames. The two last arguments aims to reduce
     the amount of data to get from the server. This method is used to have all data from the server in a same time for
     synchronizing the data.
    - using either the get-device_data or get_marker_set_data methods of the interface. In this case you don't need to
    specify the command as it was already set before when adding the device or marker set to the interface. In this case
    you can either get data directly from the server using the get from server argument or get the data from the last
    call to the get_data_from_server() method. If you want synchronized data you need to call the get_data_from_server()
    before and then get the data from the device or marker set.
Please note that if you have read once the data from the server you can't read it again. If you want to read data from
 several client you need to used the same amount of server.
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
            command=["proc_device_data", "marker_set_data"],
            nb_frame_to_get=1,
        )
        data = tcp_client.get_device_data(device_name="processed EMG", get_from_server=False)
        data_mark = tcp_client.get_marker_set_data(marker_set_name="markers", get_from_server=False)
        time.sleep(0.01)
        print(data)
        print(data_mark)
