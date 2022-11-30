import pytest
from biosiglive import (
    InterfaceType,
    ViconClient,
    PytrignoClient,
    RealTimeProcessingMethod,
    TcpClient,
    Device,
    DeviceType,
    MarkerSet,
    InverseKinematicsMethods,
)
from examples.custom_interface import MyInterface
import numpy as np
import os


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
    device = Device(
        device_type=device_type,
        nb_channels=3,
        name="my_device",
        rate=2000,
        system_rate=100,
        channel_names=["1", "2", "3"],
    )
    device.data_window = 100
    assert device.device_type == device_type
    assert device.sample == 2000 / 100
    assert device.name == "my_device"
    assert device.nb_channels == 3

    np.random.seed(50)
    i = 0
    while i != 100:
        if device_type != DeviceType.Imu:
            device.new_data = np.random.random((3, 20))
        else:
            device.new_data = np.random.random((3, 9, 20))
        device.append_data(device.new_data)
        i += 1

    raw_data = device.raw_data
    if device_type != DeviceType.Imu:
        processed_data = device.process(RealTimeProcessingMethod.ProcessGenericSignal)
        assert raw_data.shape == (3, 100)
        assert processed_data.shape == (3, 20)
    else:
        processed_data = device.process(RealTimeProcessingMethod.ProcessImu)
        assert raw_data.shape == (3, 9, 100)
        assert len(processed_data) == 3
        processed_data = device.process(RealTimeProcessingMethod.ProcessImu, squared=True)
        assert raw_data.shape == (3, 9, 100)
        assert len(processed_data.shape) == 2


def test_marker_set():
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = parent_dir + "/examples/model/Wu_Shoulder_Model_mod_wt_wrapp.bioMod"
    marker_set = MarkerSet(nb_channels=16, name="my_marker_set", rate=100, system_rate=100)
    marker_set.data_window = 100
    assert marker_set.sample == 100 / 100
    assert marker_set.name == "my_marker_set"
    assert marker_set.nb_channels == 16

    np.random.seed(50)
    i = 0
    while i != 100:
        marker_set.new_data = np.random.random((3, 16, 1))
        marker_set.append_data(marker_set.new_data)
        i += 1

    raw_data = marker_set.raw_data
    kin_data, _ = marker_set.get_kinematics(model_path=model_path, method=InverseKinematicsMethods.BiorbdKalman)
    assert raw_data.shape == (3, 16, 100)
    assert kin_data.shape == (15, 1)
