import pytest
from biosiglive import (
    InterfaceType, ViconClient, PytrignoClient, DeviceType, RealTimeProcessingMethod, TcpClient,
Device,
DeviceType)
from examples.custom_interface import MyInterface
import numpy as np


@pytest.mark.parametrize(
    "interface_type",
    [InterfaceType.Custom, InterfaceType.ViconClient, InterfaceType.PytrignoClient, InterfaceType.TcpClient],
)
def test_interface(interface_type):
    # Create interface
    interface = None
    if interface_type == InterfaceType.Custom:
        interface = MyInterface(system_rate=100)
    elif interface_type == InterfaceType.ViconClient:
        interface = ViconClient(system_rate=100, init_now=False)
    elif interface_type == InterfaceType.PytrignoClient:
        interface = PytrignoClient(system_rate=100, ip="127.0.0.1", init_now=False)
    elif interface_type == InterfaceType.TcpClient:
        interface = TcpClient(read_frequency=100, ip="127.0.0.1")

    # Add device
    interface.add_device(
        name="EMG",
        device_type=DeviceType.Emg,
        rate=2000,
        nb_channels=5,
        device_data_file_key="emg",
        data_buffer_size=2000,
        processing_method=RealTimeProcessingMethod.ProcessEmg,
        processing_window=2000,
        low_pass_filter=True,
        band_pass_filter=True,
    )
    np.testing.assert_almost_equal(len(interface.devices), 1)
    assert interface.get_device("EMG") == interface.devices[0]
    assert interface.get_device(idx=0) == interface.devices[0]

    try:
        interface.add_marker_set(name="markers", rate=100, data_buffer_size=100)
    except Exception as e:
        if interface_type == InterfaceType.PytrignoClient:
            assert e
    else:
        np.testing.assert_almost_equal(len(interface.marker_sets), 1)
        assert interface.get_marker_set("markers") == interface.marker_sets[0]
        assert interface.get_marker_set(idx=0) == interface.marker_sets[0]

@pytest.mark.parametrize("device_type", [DeviceType.Emg, DeviceType.Imu, DeviceType.Generic])
def test_devices(device_type):
    pass