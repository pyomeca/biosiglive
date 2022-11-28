# test custom interfaces
# test any interface if connected to the interface device (trigno, vicon, etc.)
import pytest
from biosiglive import (InterfaceType, ViconClient, PytrignoClient, DeviceType, RealTimeProcessingMethod, TcpClient)
from examples.biomechanical_application.custom_interface import MyInterface
import numpy as np


@pytest.mark.parametrize("interface_type", [InterfaceType.Custom, InterfaceType.ViconClient, InterfaceType.PytrignoClient, InterfaceType.TcpClient])
def test_interface(interface_type):
    # Create interface
    interface = None
    if interface_type == InterfaceType.Custom:
        interface = MyInterface(system_rate=100)
    elif interface_type == InterfaceType.ViconClient:
        interface = ViconClient(system_rate=100, init_now=False)
    elif interface_type == InterfaceType.PytrignoClient:
        interface = PytrignoClient(system_rate=100, ip="127.0.0.1", init_now=False)
    # elif interface_type == InterfaceType.TcpClient:
    #     interface = TcpClient(system_rate=100, ip="127.0.0.1", init_now=False)

    # Add device
    interface.add_device(name="EMG",
                         device_type=DeviceType.Emg,
                         rate=2000,
                         nb_channels=5,
                         device_data_file_key="emg",
                         data_buffer_size=2000,
                         processing_method=RealTimeProcessingMethod.ProcessEmg,
                         processing_window=2000,
                         low_pass_filter=True,
                         band_pass_filter=True)
    np.testing.assert_almost_equal(len(interface.devices), 1)
    try:
        interface.add_marker_set(name="markers", rate=100, data_buffer_size=100)
    except Exception as e:
        if interface_type == InterfaceType.PytrignoClient:
            assert e
    else:
        np.testing.assert_almost_equal(len(interface.marker_sets), 1)

    # connect to the device
    if interface_type != InterfaceType.Custom:
        interface.init_client()

    i = 0
    while i != 50:
        # Get data
        data = interface.get_device_data()
        if interface_type != InterfaceType.PytrignoClient:
            markers = interface.get_marker_set_data()
            kin = interface.marker_sets[0].get_kinematics(model_path="models/model.bioMod")
        processed_data = interface.devices[0].process()
        i += 1




