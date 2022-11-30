import pytest
from biosiglive import PytrignoClient, DeviceType
import numpy as np


@pytest.mark.parametrize("init", [True, False])
@pytest.mark.parametrize("get_frame", [True, False])
def test_interface(init, get_frame):
    # Create interface
    interface = PytrignoClient(system_rate=100, init_now=init)

    # Add device
    interface.add_device(name="EMG", device_type=DeviceType.Emg, nb_channels=5, device_range=(3, 8))

    if not init:
        interface.init_client()
    i = 0
    device_data = None
    while i != 50:
        # Get data
        if not get_frame:
            interface.get_frame()
        device_data = interface.get_device_data(get_frame=get_frame)
        i += 1

    np.testing.assert_almost_equal(device_data.shape[0], 5)
