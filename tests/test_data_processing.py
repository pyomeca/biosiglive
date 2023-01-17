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
        np.testing.assert_almost_equal(processed_data[:, -1], [0.156469 , 0.1365517])
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
    np.random.seed(50)
    markers_data = np.random.rand(3, 16, 1)
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
                0.4678967,
                0.5452077,
                0.6265815,
                2.3980326,
                0.7131959,
                -0.843272,
                0.4449836,
                -0.3651489,
                -1.8346828,
                -0.4779496,
                -1.9841764,
                -0.9272213,
                1.9080811,
                -2.9865801,
                2.487792,
            ],
        )
        np.testing.assert_almost_equal(q_dot[:, 0], np.zeros((15,)))

    if methods == InverseKinematicsMethods.BiorbdKalman:
        np.testing.assert_almost_equal(
            q[:, 0],
            [
                0.3967554,
                0.5319872,
                0.7010691,
                -20.2915409,
                1.360212,
                22.1051027,
                -1.4536078,
                5.3653793,
                -153.7577957,
                5.9256863,
                142.7770999,
                -82.2179062,
                2.0438505,
                43.001448,
                9.4642261,
            ],
        )
        np.testing.assert_almost_equal(
            q_dot[:, 0],
            [
                -6.83500742e00,
                1.09847707e00,
                1.57524868e00,
                -3.97775191e01,
                -3.56767334e01,
                6.14234710e01,
                4.47125570e01,
                1.27014192e02,
                -1.10512674e02,
                -4.87175834e-01,
                -2.81411720e01,
                1.01959841e02,
                5.55270630e01,
                6.81543760e02,
                1.93372073e01,
            ],
            decimal=6,
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
