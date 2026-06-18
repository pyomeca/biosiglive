from enum import Enum


class AvantiAdapter(Enum):
    """
    Return mode of sensor
    """

    Gogniometer = "23"
    NONE = "O"


class AvantiSensor:
    def __init__(self, adapter: AvantiAdapter = AvantiAdapter.NONE):
        self.emg_port = 50043
        self.aux_port = 50044
        self.type = adapter.value


class LegacySensor:
    def __init__(self):
        self.emg_port = 50041
        self.aux_port = 50042
        self.type = "A"


class SensorType(Enum):
    AVANTI = "Avanti"
    LEGACY = "Legacy"
    # ALL = "All"
    # Avanti_all = "Avanti_all"
    # Avanti_emg = "Avanti_emg"
    # Avanti_aux = "Avanti_aux"
    # Legacy_all = "Legacy_all"
    # Legacy_emg = "Legacy_emg"
    # Legacy_aux = "Legacy_aux"
