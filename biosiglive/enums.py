from enum import Enum


class InterfaceType(Enum):
    ViconClient = "vicon_client"
    PytrignoClient = "pytrigno_client"
    TcpClient = "tcp_client"
    Custom = "custom"


class DeviceType(Enum):
    Emg = "emg"
    Imu = "imu"
    Generic = "generic"
    ForcePlate = "force_plate"


class MarkerType(Enum):
    Labeled = "labeled"
    Unlabeled = "unlabeled"


class PlotType(Enum):
    Curve = "curve"
    ProgressBar = "progress_bar"
    Skeleton = "skeleton"
    Scatter3D = "scatter3d"


class InverseKinematicsMethods(Enum):
    BiorbdKalman = "biorbd_kalman"
    BiorbdLeastSquare = "biorbd_least_square"
    Custom = "custom"


class RealTimeProcessingMethod(Enum):
    ProcessEmg = "process_emg"
    ProcessImu = "process_imu"
    CalibrationMatrix = "calibration_matrix"
    GetPeaks = "get_peaks"
    Custom = "custom"
    ProcessGenericSignal = "process_generic_signal"


class OfflineProcessingMethod(Enum):
    ProcessEmg = "process_emg"
    CalibrationMatrix = "calibration_matrix"
    ComputeMvc = "compute_mvc"
