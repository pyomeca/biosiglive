"""
This example show how to use the StreamData class to stream data from an interface, process it using several process to
finally send the data through a server to a client. Each task in a separate process to allow the streaming and the
 processing to be done in real-time. Please note that for now only a number equal to the number of cores of the computer
  is supported.
First an interface is created and device and marker_set are added to it (please refer to EMG_streming.py and
marker_streaming.py for more details).
Then a StreamData object is created. The StreamData object takes as argument the targeted frequency at which the data will
be streamed. Then the interface is added to the StreamData object. If the user want to start a server to
disseminate the data a server can be added to the StreamData object specifying the ip address and the port and the
data buffer for the device and the marker set. The data buffer is the number of frame that will be stored in the server,
it will be use if the client need a specific amount of data.
Then the streaming will be started with all the data streaming, processing and the server in seperate process. If no
 processing method is specified the data will be streamed as it is and no additional process will be started. A file can
  be specified to save the data. The data will be saved in a *.bio file at each loop of the data streaming by default or
  at the save frequency specified in the start method.
Please note that it is not yet possible to plot the data in real-time.
"""

from custom_interface import MyInterface
from biosiglive import (
    ViconClient,
    PytrignoClient,
    StreamData,
    DeviceType,
    InverseKinematicsMethods,
    RealTimeProcessingMethod,
    InterfaceType,
)

try:
    import biorbd
except ModuleNotFoundError:
    biorbd_package = False

try:
    from vicon_dssdk import ViconDataStream as VDS
except ModuleNotFoundError:
    vicon_package = False


if __name__ == "__main__":
    server_ip = "127.0.0.1"
    server_port = 50000
    interface_type = InterfaceType.Custom

    if interface_type == InterfaceType.Custom:
        interface = MyInterface(system_rate=100, data_path="abd.bio")
    elif interface_type == InterfaceType.ViconClient:
        interface = ViconClient(system_rate=100)
    elif interface_type == InterfaceType.PytrignoClient:
        interface = PytrignoClient(system_rate=100, ip="127.0.0.1")
    else:
        raise ValueError("The type of interface is not valid.")

    model_path = "model/Wu_Shoulder_Model_mod_wt_wrapp.bioMod"
    nb_electrode = 5
    interface.add_device(
        name="EMG",
        device_type=DeviceType.Emg,
        rate=2000,
        nb_channels=nb_electrode,
        device_data_file_key="emg",
        data_buffer_size=2000,
        processing_method=RealTimeProcessingMethod.ProcessEmg,
        processing_window=1,
        moving_average_window=1,
        low_pass_filter=False,
        band_pass_filter=True,
        normalization=False,
    )
    interface.get_device("EMG").process_method = RealTimeProcessingMethod.ProcessEmg
    interface.get_device("EMG").process_method_kwargs = {
        "processing_window": 2000,
        "low_pass_filter": False,
        "band_pass_filter": True,
    }

    interface.add_marker_set(
        name="markers",
        rate=100,
        data_buffer_size=100,
        kinematics_method=InverseKinematicsMethods.BiorbdKalman,
        model_path=model_path,
        marker_data_file_key="markers",
        nb_markers=16,
    )
    data_streaming = StreamData(stream_rate=100)
    data_streaming.add_interface(interface)
    data_streaming.add_server(server_ip, server_port, device_buffer_size=20, marker_set_buffer_size=1)
    data_streaming.start(save_streamed_data=True, save_path="data_streamed")
