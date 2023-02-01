import pytest
from biosiglive import (
    RealTimeProcessing,
    OfflineProcessing,
    RealTimeProcessingMethod,
    OfflineProcessingMethod,
    MskFunctions,
    InverseKinematicsMethods,
)
import numpy as np
import os


def custom_function(new_sample):
    return new_sample


@pytest.mark.parametrize(
    "method",
    [
        RealTimeProcessingMethod.CalibrationMatrix,
        RealTimeProcessingMethod.ProcessEmg,
        RealTimeProcessingMethod.GetPeaks,
        RealTimeProcessingMethod.ProcessGenericSignal,
        RealTimeProcessingMethod.ProcessImu,
        RealTimeProcessingMethod.Custom,
    ],
)
def test_real_time_processing(method):
    np.random.seed(50)
    data = np.random.rand(2, 20)
    data_3d = np.random.rand(4, 3, 20)
    processing = RealTimeProcessing(data_rate=2000, processing_window=1000)
    processed_data = None
    nb_peaks = None
    i = 0
    while i != 150:
        if method == RealTimeProcessingMethod.CalibrationMatrix:
            processed_data = processing.calibration_matrix(data, np.random.rand(2, 2))
        elif method == RealTimeProcessingMethod.ProcessEmg:
            processed_data = processing.process_emg(data)
        elif method == RealTimeProcessingMethod.GetPeaks:
            data_tmp = processing.get_peaks(data, threshold=0.1)
            processed_data = data_tmp[1]
            nb_peaks = data_tmp[0]
        elif method == RealTimeProcessingMethod.ProcessGenericSignal:
            processed_data = processing.process_generic_signal(data)
        elif method == RealTimeProcessingMethod.ProcessImu:
            processed_data = processing.process_imu(data_3d)
        elif method == RealTimeProcessingMethod.Custom:
            processed_data = processing.custom_processing(custom_function, data)

        if method == RealTimeProcessingMethod.ProcessImu:
            np.testing.assert_almost_equal(len(processed_data.shape), 3)
        else:
            np.testing.assert_almost_equal(len(processed_data.shape), 2)
        i += 1

    if method == RealTimeProcessingMethod.CalibrationMatrix:
        np.testing.assert_almost_equal(processed_data[:, -1], [0.4959683, 0.3568878])
    elif method == RealTimeProcessingMethod.ProcessEmg:
        np.testing.assert_almost_equal(processed_data[:, -1], [0.156469, 0.1365517])
    elif method == RealTimeProcessingMethod.GetPeaks:
        np.testing.assert_almost_equal(processed_data[:, -1], [0.0, 0.0])
        np.testing.assert_almost_equal(nb_peaks, 100)
    elif method == RealTimeProcessingMethod.ProcessGenericSignal:
        np.testing.assert_almost_equal(processed_data[:, -1], [0.0164153, 0.0218168])
    elif method == RealTimeProcessingMethod.Custom:
        np.testing.assert_almost_equal(processed_data[:, -1], [0.3910874, 0.2220394])
    elif method == RealTimeProcessingMethod.ProcessImu:
        np.testing.assert_almost_equal(processed_data[:, 0, -1], [0.5367742, 0.477345, 0.3982464, 0.464923])


@pytest.mark.parametrize("method", [OfflineProcessingMethod.ProcessEmg, OfflineProcessingMethod.ComputeMvc])
def test_offline_processing(method):
    np.random.seed(50)
    data = np.random.rand(2, 4000)
    processing = OfflineProcessing(data_rate=2000, processing_window=1000)

    if method == OfflineProcessingMethod.ProcessEmg:
        processed_data = processing.process_emg(data)
        np.testing.assert_almost_equal(processed_data[:, 0], [0.1021941, 0.0959892])
    elif method == OfflineProcessingMethod.ComputeMvc:
        processed_data = processing.compute_mvc(2, data, 2000)
        np.testing.assert_almost_equal(processed_data[:], [0.8815387, 0.873678])


@pytest.mark.parametrize(
    "methods",
    [
        InverseKinematicsMethods.BiorbdLeastSquare,
        InverseKinematicsMethods.BiorbdKalman,
    ],
)
def test_inverse_kinematics_methods(methods):
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    markers_data = np.ones((3, 16, 1)) * 0.1
    model_path = parent_dir + "/examples/model/Wu_Shoulder_Model_mod_wt_wrapp.bioMod"

    msk_function = MskFunctions(model=model_path)
    q, q_dot = None, None
    i = 0
    while i != 15:
        q, q_dot = msk_function.compute_inverse_kinematics(markers_data, method=methods)
        i += 1

    if methods == InverseKinematicsMethods.BiorbdLeastSquare:
        np.testing.assert_almost_equal(
            q[:, 0],
            [
                0.0859061,
                0.1650248,
                0.1019853,
                -2.3876092,
                1.3106894,
                2.7743532,
                -3.8583198,
                -0.2704209,
                2.0373899,
                0.1503456,
                0.9030997,
                0.8522711,
                -1.6218697,
                0.4328726,
                -3.10159,
            ],
            decimal=4,
        )
        np.testing.assert_almost_equal(q_dot[:, 0], np.zeros((15,)))

    if methods == InverseKinematicsMethods.BiorbdKalman:
        np.testing.assert_almost_equal(
            q[:, 0],
            [
                0.1641379,
                0.1138498,
                0.1144206,
                1.2445316,
                0.0922355,
                -0.9251613,
                -3.9271956,
                -2.9052048,
                -12.9825494,
                -2.8963511,
                -14.2619434,
                4.0001431,
                -4.6817521,
                4.2250031,
                -3.1007907,
            ],
            decimal=4,
        )
        np.testing.assert_almost_equal(
            q_dot[:, 0],
            [
                -0.013688,
                -0.0174,
                0.054321,
                3.4537,
                0.611992,
                0.115344,
                -5.570652,
                -3.603674,
                0.370434,
                -1.234856,
                -0.491976,
                -0.033802,
                -0.031005,
                0.610965,
                0.03146,
            ],
            decimal=4,
        )


def test_forward_kinematics():
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    np.random.seed(50)
    q_data = np.random.rand(15, 1)
    model_path = parent_dir + "/examples/model/Wu_Shoulder_Model_mod_wt_wrapp.bioMod"

    msk_function = MskFunctions(model=model_path)
    markers = None
    i = 0
    while i != 15:
        markers = msk_function.compute_direct_kinematics(q_data)
        i += 1

    np.testing.assert_almost_equal(
        markers[0, :, 0],
        [
            0.5047011,
            0.6690358,
            0.4138976,
            0.58448,
            0.5091725,
            0.4676433,
            0.6165203,
            0.4549606,
            0.5009109,
            0.7444251,
            0.787134,
            0.5910698,
            0.6242104,
            0.9334877,
            0.7964393,
            0.8911985,
        ],
    )
