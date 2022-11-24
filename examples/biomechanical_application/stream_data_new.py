
from custom_interface import MyInterface
from biosiglive import (
    ViconClient,
    PytrignoClient,
    StreamData,
    LivePlot,
    DeviceType,
    InverseKinematicsMethods,
    RealTimeProcessingMethod,
    InterfaceType,
    PlotType
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

    # Initialize interface and add device and markers
    if interface_type == InterfaceType.Custom:
        interface = MyInterface(system_rate=100)
    elif interface_type == InterfaceType.ViconClient:
        interface = ViconClient(system_rate=100)
    elif interface_type == InterfaceType.PytrignoClient:
        interface = PytrignoClient(system_rate=100, ip="127.0.0.1")
    else:
        raise ValueError("The type of interface is not valid.")

    model_path = "model/Wu_Shoulder_Model_mod_wt_wrapp.bioMod"
    nb_electrode = 5
    interface.add_device(name="EMG",
                         device_type=DeviceType.Emg,
                         rate=2000,
                         nb_channels=nb_electrode,
                         data_buffer_size=2000,
                         process_method=RealTimeProcessingMethod.ProcessEmg,
                         processing_window=2000,
                         low_pass_filter=False,
                         band_pass_filter=True,
                         normalization=False)
    interface.get_device("EMG").process_method = RealTimeProcessingMethod.ProcessEmg
    interface.get_device("EMG").process_method_kwargs = {"processing_window": 2000, "low_pass_filter": False, "band_pass_filter": True,}

    interface.add_marker_set(name="markers", marker_names=["a", "b"], rate=100, data_buffer_size=100,
                             kinematics_method=InverseKinematicsMethods.BiorbdLeastSquare,
                             model=model_path)

    # Initialize plot
    emg_plot = LivePlot(name="emg", rate=50, plot_type=PlotType.ProgressBar, nb_subplots=nb_electrode)
    emg_plot.init(plot_windows=[interface.devices[0].rate, interface.devices[0].rate, interface.system_rate])
    kin_plot = LivePlot(name="Kin", rate=50, plot_type="skeleton", nb_subplots=1)
    kin_plot.init(msk_model=model_path, show_muscles=False, show_floor=False)

    # Initialize StreamData
    data_streaming = StreamData(stream_rate=100)
    data_streaming.add_interface(interface)
    data_streaming.add_plot([emg_plot, kin_plot], data_to_plot=["EMG", "markers"], raw=[False, False], multiprocess=False)
    data_streaming.add_server(server_ip, server_port, device_buffer_size=20, marker_set_buffer_size=1)
    data_streaming.start(save_streamed_data=True, save_path="data_streamed")
