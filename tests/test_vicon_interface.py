import pytest
from biosiglive import ViconClient, DeviceType
import numpy as np


@pytest.mark.parametrize("init", [True, False])
@pytest.mark.parametrize("get_frame", [True, False])
def test_interface(init, get_frame):
    # Create interface
    interface = ViconClient(system_rate=100, init_now=init)

    # Add device
    interface.add_device(
        name="EMG",
        device_type=DeviceType.Emg,
        nb_channels=5,
    )
    interface.add_marker_set(nb_markers=5, name="markers", rate=100)

    if not init:
        interface.init_client()
    i = 0
    while i != 50:
        # Get data
        if not get_frame:
            interface.get_frame()
        device_data = interface.get_device_data(get_frame=get_frame)
        markers_data = interface.get_marker_set_data(get_frame=get_frame)
        latency = [interface.get_latency()]
        frame_number = [interface.get_frame_number()]

        i += 1

    np.testing.assert_almost_equal(device_data.shape[0], 5)
    np.testing.assert_almost_equal(markers_data.shape[1], 10)
    np.testing.assert_almost_equal(len(latency), 1)
    np.testing.assert_almost_equal(len(frame_number), 1)
