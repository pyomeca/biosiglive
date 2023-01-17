from biosiglive import StreamData, TcpClient, DeviceType, RealTimeProcessingMethod, InverseKinematicsMethods
from examples.custom_interface import MyInterface
from threading import Thread
import os
import time


# TODO fix this test so the program end when it is done
def test_tcp_client():
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    server_ip = "127.0.0.1"
    server_port = 50000
    model_path = parent_dir + "/examples/model/Wu_Shoulder_Model_mod_wt_wrapp.bioMod"
    nb_electrode = 5
    interface = MyInterface(system_rate=100, data_path=parent_dir + "/examples/abd.bio")
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

    interface.add_marker_set(
        name="markers",
        rate=100,
        data_buffer_size=100,
        kinematics_method=InverseKinematicsMethods.BiorbdKalman,
        model_path=model_path,
        marker_data_file_key="markers",
        nb_markers=16,
        unit="mm",
    )

    def stream_data(interface):
        stream_data = StreamData(stream_rate=100)
        stream_data.add_interface(interface)
        stream_data.add_server(server_ip, server_port)
        stream_data.start()

    server_thread = Thread(target=stream_data, args=(interface,))
    server_thread.start()
    time.sleep(1)
    # create a client
    tcp_client = TcpClient(server_ip, server_port, read_frequency=100)
    i = 0
    command = ["proc_device_data", "marker_set_data", "raw_device_data", "kinematics_data"]
    data = {}
    while i != 5:
        data = tcp_client.get_data_from_server(command=command, nb_frame_to_get=2, down_sampling={"emg": 20})
        i += 1

    shapes_0 = [5, 3, 5, 15]
    shapes_1 = [2, 16, 2, 2]
    for c, com in enumerate(command):
        assert com == list(data.keys())[c]
        assert data[com].shape[0] == shapes_0[c]
        assert data[com].shape[1] == shapes_1[c]
        if len(data[com].shape) == 3:
            assert data[com].shape[2] == 1
