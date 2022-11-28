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


def custom_function(new_sample):
    return new_sample


@pytest.mark.parametrize("method", [RealTimeProcessingMethod.CalibrationMatrix,
                                    RealTimeProcessingMethod.ProcessEmg,
                                    RealTimeProcessingMethod.GetPeaks, RealTimeProcessingMethod.ProcessGenericSignal,
                                    RealTimeProcessingMethod.ProcessImu,
                                    RealTimeProcessingMethod.Custom])
def test_real_time_processing(method):
    np.random.seed(50)
    data = np.random.rand(2, 4000)
    data_3d = np.random.rand(6, 2, 4000)
    processing = RealTimeProcessing(data_rate=2000, processing_window=1000)
    processed_data = None
    nb_peaks = None
    i = 0
    while i != 15:
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
        np.testing.assert_almost_equal(processed_data[:, 0], [0.6411369, 0.7392561])
    elif method == RealTimeProcessingMethod.ProcessEmg:
        np.testing.assert_almost_equal(processed_data[:, 0], [0.1388693, 0.1380089])
    elif method == RealTimeProcessingMethod.GetPeaks:
        np.testing.assert_almost_equal(processed_data[:, 0], [0.0, 0.0])
        np.testing.assert_almost_equal(nb_peaks, 694)
    elif method == RealTimeProcessingMethod.ProcessGenericSignal:
        np.testing.assert_almost_equal(processed_data[:, 0], [0.1021941, 0.0959892])
    elif method == RealTimeProcessingMethod.Custom:
        np.testing.assert_almost_equal(processed_data[:, 0], [0.4946016, 0.7849796])
    elif method == RealTimeProcessingMethod.ProcessImu:
        np.testing.assert_almost_equal(processed_data[:, 0, 0], [0.5, 0.5])


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


@pytest.mark.parametrize("method", [InverseKinematicsMethods.BiorbdLeastSquare, InverseKinematicsMethods.BiorbdKalman,
    InverseKinematicsMethods.Custom])
def test_inverse_kinematics_methods(method):
    pass


def test_forward_kinematics(method):
    pass

# check has changed
# check random processing
# check custom processing
# check msk fucntions
# check mvc function
# check any processing function