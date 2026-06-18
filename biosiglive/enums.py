"""
Groups the different enums used in the program. it is a good place to start to check what's available.
"""

from enum import Enum


class InterfaceType(Enum):
    """
    The different types of interfaces that can be used.
    """

    ViconClient = "vicon_client"
    PytrignoClient = "pytrigno_client"
    TcpClient = "tcp_client"
    Custom = "custom"


class DeviceType(Enum):
    """
    The different types of devices that can be used.
    """

    Emg = "emg"
    Imu = "imu"
    DelsysGogniometer = "delsys_gogniometer"
    Generic = "generic"
    ForcePlate = "force_plate"


class MarkerType(Enum):
    """
    The different types of markers that can be used.
    """

    Labeled = "labeled"
    Unlabeled = "unlabeled"


class PlotType(Enum):
    """
    The different types of plots that can be used.
    """

    Curve = "curve"
    ProgressBar = "progress_bar"
    Skeleton = "skeleton"
    Scatter3D = "scatter3d"


class InverseKinematicsMethods(Enum):
    """
    The different types of inverse kinematics methods that can be used.
    """

    BiorbdKalman = "biorbd_kalman"
    BiorbdLeastSquare = "biorbd_least_square"
    Custom = "custom"


class RealTimeProcessingMethod(Enum):
    """
    The different types of real time processing methods that can be used.
    """

    ProcessEmg = "process_emg"
    ProcessImu = "process_imu"
    CalibrationMatrix = "calibration_matrix"
    GetPeaks = "get_peaks"
    Custom = "custom"
    ProcessGenericSignal = "process_generic_signal"


class OfflineProcessingMethod(Enum):
    """
    The different types of offline processing methods that can be used.
    """

    ProcessEmg = "process_emg"
    CalibrationMatrix = "calibration_matrix"
    ComputeMvc = "compute_mvc"
