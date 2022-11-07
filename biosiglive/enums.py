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


class InverseKinematicsMethods(Enum):
    BiorbdKalman = "biorbd_kalman"
    BiorbdLeastSquare = "biorbd_least_square"
    Custom = "custom"
