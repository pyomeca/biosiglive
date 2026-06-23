"""
This file contains a wrapper for the python qualisys SDK.
"""
from .param import *
from typing import Union
from .generic_interface import GenericInterface
from ..enums import InverseKinematicsMethods, InterfaceType
import xml.etree.ElementTree as ET
import numpy as np

try:
    import asyncio
    import qtm_rt
except ModuleNotFoundError:
    pass


class QualisysClient(GenericInterface):
    """
    Class for interfacing with the Qualisys system.
    """

    def __init__(self, system_rate: int, ip: str = "192.168.254.1", port: int = 22224, init_now=True):
        """
        Initialize the QualisysClient class.

        Parameters
        ----------
        system_rate: int
            Streaming rate of the Qualisys software.
        ip: str
            IP address of the Qualisys software.
        port: int
            Port of the Qualisys software.
        init_now: bool
            Whether to initialize the client now.
            Usefull if you want to pickle the interface as the qualisys SDK is not pickable (swig).
        """
        super(QualisysClient, self).__init__(ip=ip, system_rate=system_rate, interface_type=InterfaceType.QualisysClient)
        self.address = ip
        self.port = port
        self.qualisys_client = None
        self.acquisition_rate = None
        self.system_rate = system_rate
        self.devices = []
        self.forces = []
        self.imu = []
        self.marker_sets = []
        self.force_plates = []
        self.is_frame = False
        self.is_initialized = False
        self.component = []
        if init_now:
            asyncio.run(self._init_client())


    async def _init_client(self):
        """
        Initialize the qualisys client.

        """
        print(f"Connection to Qualisys DataStreamSDK at : {self.address} ...")
        self.Connect = await qtm_rt.connect("192.168.254.1")
        if self.Connect is None:
            print("Error","Failed to connect")
            return

        print("Connected to Qualisys")
        self.is_initialized = True
        self.qualisys_client = True

        xml_general = await self.Connect.get_parameters(parameters=["general"])
        generalinfo = ET.fromstring(xml_general) 
        frame_rate = int(generalinfo.find('.//Frequency').text)
        if self.system_rate != frame_rate:
            raise ValueError(
                f"qualisys system rate ({frame_rate}) does not match the system rate "
                f"({self.system_rate})."
            )

    @classmethod
    async def create(cls, system_rate=100, ip="192.168.254.1", port=22224):
        self = cls(system_rate, ip, port, init_now=False)
        await self._init_client()
        return self

    async def add_device(
        self,
        nb_channels: int,
        device_type: Union[DeviceType, str] = DeviceType.ForcePlate,
        data_buffer_size: int = None,
        name: str = None,
        rate: float = 2000,
        device_range: tuple = None,
        processing_method: Union[RealTimeProcessingMethod, OfflineProcessingMethod] = None,
        **process_kwargs,
    ):
        """
        Add a device to the qualisys system.

        Parameters
        ----------
        nb_channels: int
            Number of channels of the device.
        device_type: Union[DeviceType, str]
            Type of the device.
        data_buffer_size: int
            Size of the buffer for the device.
        name: str
            Name of the device.
        rate: float
            Rate of the device.
        device_range: tuple
            Range of the device.
        processing_method : Union[RealTimeProcessingMethod, OfflineProcessingMethod]
            Method used to process the data.
        **process_kwargs
            Keyword arguments for the processing method.
        """
        device_tmp = self._add_device(
            nb_channels, device_type, name, rate, device_range, processing_method, **process_kwargs
        )
        device_tmp.interface = self.interface_type
        if self.qualisys_client:
            if DeviceType.ForcePlate:
                self.component.append('Force')
                xml_force = await self.Connect.get_parameters(parameters=["force"])
                ForceInfo = ET.fromstring(xml_force)
                device_tmp.infos = ForceInfo
                device_tmp.data_windows = data_buffer_size
                device_tmp.name = [label.find('Name').text for label in ForceInfo.findall(".//Plate")]
                #device_tmp.infos.unit = ForceInfo.find('.//Unit_Force').text
                device_tmp.rate = ForceInfo.find('.//Frequency').text
                self.forces.append(device_tmp)
            else:
                self.component.append('Analog')
                xml_analog = await self.Connect.get_parameters(parameters=["analog"])
                device_tmp.infos = ET.fromstring(xml_analog)
                device_tmp.data_windows = data_buffer_size
                self.devices.append(device_tmp)
        else: 
            device_tmp.infos = None
            device_tmp.data_windows = data_buffer_size
            self.devices.append(device_tmp)

    """
    async def add_forceplate(
        self,
        data_buffer_size: int = None,
        name: str = None,
        rate: float = 1000,
        unit: str = "N",
        force_range: tuple = None,
        processing_method: Union[RealTimeProcessingMethod, OfflineProcessingMethod] = None,
        **process_kwargs,
    ):
    """
    """
        Add a device to the qualisys system.

        Parameters
        ----------
        nb_channels: int
            Number of channels of the device.
        device_type: Union[DeviceType, str]
            Type of the device.
        data_buffer_size: int
            Size of the buffer for the device.
        name: str
            Name of the device.
        rate: float
            Rate of the device.
        device_range: tuple
            Range of the device.
        processing_method : Union[RealTimeProcessingMethod, OfflineProcessingMethod]
            Method used to process the data.
        **process_kwargs
            Keyword arguments for the processing method.
    """
    """
        forceplate_tmp = self._add_forceplate(
            nb_forceplate=nb_forceplate,
            name=name,
            marker_names=plate_names,
            rate=rate,
            **kin_method_kwargs,
            )

        forceplate_tmp.interface = self.interface_type
        if self.qualisys_client:
            xml_force = await self.Connect.get_parameters(parameters=["force"])
            forceplate_tmp.infos = ET.fromstring(xml_force)
            forceplate_tmp.unit_forceplate = ForceInfo.find('.//Unit_Force').text
            forceplate_tmp.forces_names = [label.find('Name').text for label in ForceInfo.findall(".//Plate")]
            self.component.append('force')
        else:
            forceplate_tmp.infos = None
            forceplate_tmp.forces_names = name

        forceplate_tmp.data_windows = data_buffer_size
        self.forces.append(forceplate_tmp)
    """

    async def add_marker_set(
        self,
        nb_markers: int,
        name: str = None,
        data_buffer_size: int = None,
        marker_names: Union[str, list] = None,
        rate: float = 100,
        unlabeled: bool = False,
        subject_name: str = None,
        kinematics_method: InverseKinematicsMethods = None,
        **kin_method_kwargs,
    ):
        """
        Add markers set to stream from the qualisys system.

        Parameters
        ----------
        nb_markers: int
            Number of markers.
        name: str
            Name of the markers set.
        data_buffer_size: int
            Size of the buffer for the markers set.
        marker_names: Union[list, str]
            List of markers names.
        rate: int
            Rate of the markers set.
        unlabeled: bool
            Whether the markers set is unlabeled.
        subject_name: str
            Name of the subject. If None, the subject will be the first one in Nexus.
        kinematics_method: InverseKinematicsMethods
            Method used to compute the kinematics.
        **kin_method_kwargs
            Keyword arguments for the kinematics method.
        """
        if len(self.marker_sets) != 0:
            raise ValueError("Only one marker set can be added for now.")

        markers_tmp = self._add_marker_set(
            nb_markers=nb_markers,
            name=name,
            marker_names=marker_names,
            rate=rate,
            unlabeled=unlabeled,
            kinematics_method=kinematics_method,
            **kin_method_kwargs,
        )
        if self.qualisys_client:
            markers_tmp.subject_name = subject_name #a changer quand je saurais comment recuperer nom du sujet avec qualisys
            xlm_mks = await self.Connect.get_parameters(parameters=['3d'])
            MksInfo = ET.fromstring(xlm_mks)
            markers_tmp.marker_names =[label.find('Name').text for label in MksInfo.findall(".//Label")]
            self.marker_names=markers_tmp.marker_names
            self.component.append('3d')
            self.component.append('3dnolabels')
        else:
            markers_tmp.subject_name = subject_name
            markers_tmp.marker_names = marker_names
        markers_tmp.data_windows = data_buffer_size
        self.marker_sets.append(markers_tmp)

    async def get_force_plate_data(
            self, forceplate_name: Union[str, list] = "all", get_frame: bool = True, packet=None
    ):
        if len(self.forces) == 0:
            raise ValueError("No force has been added to the qualisys system.")
        if not self.is_initialized:
            raise RuntimeError("Qualisys client is not initialized.")
        if get_frame:
            packet.framenumber

        headerf, forcesdata = packet.get_force()
        all_forces_data=[]
        # Initialisation des données dynamiques
        nb_pf = len(forcesdata)  # Nombre de plaques de force
        collected_data = []  # Liste dynamique pour collecter les données valides
        PFForce = forcesdata[0][1]  # Données pour une plaque


        nb_frames = max(len(forcesdata[0][1]), len(forcesdata[1][1]))  # Nombre de frames pour cette plaque
        # Collecte des données
        for platenum in range(nb_pf):
            PFForce = forcesdata[platenum][1]
            # Temporaire pour cette plaque
            plate_data =np.empty((9, nb_frames))
            plate_data[:]=np.nan
            for frame_idx, data_tmp in enumerate(PFForce):
                # Récupérer les données et remplir la matrice temporaire
                plate_data[:, frame_idx] = [
                    data_tmp.x, data_tmp.y, data_tmp.z,
                    data_tmp.x_m, data_tmp.y_m, data_tmp.z_m,
                    data_tmp.x_a, data_tmp.y_a, data_tmp.z_a
                ]

            collected_data.append(plate_data)

            # Concaténation des données valides uniquement pour obtenir [9 * nb_pf, nb_frame]
        all_forces_data = np.concatenate(collected_data, axis=0)

        return all_forces_data


    def get_device_data(
        self, device_name: Union[str, list] = "all", channel_idx: Union[int, list] = (), get_frame: bool = True
    ):
        """
        Get the device data from qualisys.

        Parameters
        ----------
        device_name: str or list
            Name of the device or list of devices names.
        channel_idx: Union[int, str]
            Index of the channel to return.
        get_frame: bool
            Whether to get a new frame from the qualisys system.

        Returns
        -------
        device_data: list
            All asked device data.
        """
        if len(self.devices) == 0:
            raise ValueError("No device has been added to the qualisys system.")
        if not self.is_initialized:
            raise RuntimeError("Qualisys client is not initialized.")
        if get_frame:
            self.packet.framenumber
        all_device_data = []
        if not isinstance(device_name, list):
            device_name = [device_name]
        if channel_idx and not isinstance(channel_idx, list):
            channel_idx = [channel_idx]
        device_data = []
        headerdevice, Devicevalue = self.packet.get_analog()
        for d, device in enumerate(self.devices):
            if device_name[0] == "all" or device.name in device_name:
                device.new_data = np.zeros((device.nb_channels, device.sample))
                count = 0
                for output_name, channel_name, unit in device.infos:
                    data_tmp, _ = self.packet.get_analog(device.name, output_name, channel_name)
                    device.new_data[count, :] = data_tmp
                    device.channel_names.append(channel_name)
                    count += 1
                    if count == device.nb_channels:
                        break
                if channel_idx:
                    device_data = np.zeros((len(channel_idx), device.sample))
                    for idx in range(device.nb_channels):
                        if idx in channel_idx:
                            device_data[channel_idx.index(idx), :] = device.new_data[idx, :]
                device_data = device_data if channel_idx else device.new_data
                device.append_data(device.new_data)
                all_device_data.append(device_data)
        if len(all_device_data) == 1:
            return all_device_data[0]
        return all_device_data

    async def get_marker_set_data(
        self, subject_name: Union[str, list] = None, marker_names: Union[str, list] = None, get_frame: bool = True, packet = None
    ):
        """
        Get the markers data from qualisys.

        Parameters
        ----------
        subject_name: Union[str, list]
            Name of the subject. If None, the subject will be the first one in Nexus.
        marker_names: Union[str, list]
            List of markers names.
        get_frame: bool
            Whether to get a new frame or not.

        Returns
        -------
        markers_data: list
            All asked markers data.
        """

        if len(self.marker_sets) == 0:
            raise ValueError("No marker set has been added to the qualisys system.")
        if not self.is_initialized:
            raise RuntimeError("qualisys client is not initialized.")
        if get_frame:
            packet.framenumber
        if subject_name and isinstance(subject_name, list):
            subject_name = [subject_name]
        if marker_names and isinstance(marker_names, list):
            marker_names = [marker_names]
        #occluded = []
        all_markers_data = []
        #all_occluded_data = []
        if subject_name:
            marker_sets = [None] * len(subject_name)
            for s, marker_set in enumerate(self.marker_sets):
                if marker_set.subject_name in subject_name:
                    marker_sets[subject_name.index(marker_set.subject_name)] = marker_set
            if marker_sets == [None]:
                raise RuntimeError("No subject of this name.")
        else:
            marker_sets = self.marker_sets

        for markers in marker_sets:
            markers.new_data = np.zeros((3, len(markers.marker_names), markers.sample))
            count = 0
            header, allmarkers_data_tmp= packet.get_3d_markers() #TO check
            #headernolabel, allmarkersnolabel_data_tmp= packet.get_3d_markers_no_label
            for m, marker_name in enumerate(markers.marker_names):
                markers_data_tmp=allmarkers_data_tmp[m][:]
                markers.new_data[:, m, :] = np.array(markers_data_tmp)[:, np.newaxis]
                #occluded.append(occluded_tmp)

            all_markers_data.append(markers.new_data)
            markers.append_data(markers.new_data)
            #all_occluded_data.append(occluded)
        if len(all_markers_data) == 1:
            #print(packet.framenumber)
            #print(packet.timestamp)
            return all_markers_data[0], self.marker_names #, all_occluded_data[0]
        return all_markers_data.marker_names  #, all_occluded_data

    async def init_client(self):
        """
        Initialize the Qualisys client if it is not already initialized.
        This function has to be called before get frame from interface.
        """
        if self.is_initialized:
            raise RuntimeError("Qualisys client is already initialized.")
        else:
            self._init_client()
            xlm_analog = await self.Connect.get_parameters(parameters=['analog'])
            AnalogInfo = ET.fromstring(xlm_analog)
            for d, device in enumerate(self.devices):
                if not device.infos:
                    device.infos = self.qualisys_client.GetDeviceOutputDetails(device.name) #a changer
            
            xlm_3d = await self.Connect.get_parameters(parameters=['3d'])
            MksInfo = ET.fromstring(xlm_3d)
            nom_mks = [label.find('Name').text for label in MksInfo.findall(".//Label")]
            for m, marker_set in enumerate(self.marker_sets):
                if not marker_set.markers_names:
                    marker_set.markers_names = [label.find('Name').text for label in MksInfo.findall(".//Label")]
                if not marker_set.subject_name:
                    marker_set.subject_name = [label.find('Name').text for label in MksInfo.findall(".//Label")]

    async def get_latency(self) -> float:
        """
        Get the latency between the qualisys system and the qualisys SDK.

        Returns
        -------
        latency: float
            Latency between the qualisys system and the qualisys SDK.
        """
        if not self.is_initialized:
            raise RuntimeError("qualisys client is not initialized.")
        self.qualisys_client = 0 #voir comment obtenir la latence
        return self.qualisys_client

    async def get_frame(self) -> bool:
        """
        Get a new frame from the qualisys system.

        Returns
        -------
        bool
            True if there is a frame, False otherwise.
        """
        """
        if not self.is_initialized:
            raise RuntimeError("qualisys client is not initialized.")
        self.is_frame = self.qualisys_client.GetFrame()
        while self.is_frame is not True:
            self.is_frame = self.qualisys_client.GetFrame()
        return self.is_frame
        """ #TODO

    async def get_frame_number(self) -> int:
        """
        Get the last frame number.

        Returns
        -------
        frame_number: int
            Last frame number.
        """
        if not self.is_initialized:
            raise RuntimeError("qualisys client is not initialized.")
        return self.packet.framenumber

    async def get_kinematics_from_markers(
        self,
        marker_set_name: str,
        model_path: str = None,
        method: Union[InverseKinematicsMethods, str] = InverseKinematicsMethods.BiorbdLeastSquare,
        custom_func: callable = None,
        **kwargs,
    ):
        """
        Get the kinematics from markers.

        Parameters
        ----------
        marker_set_name: str
            name of the markerset.
        model_path: str
            biorbd model of the kinematics.
        method: str
            Method to use to get the kinematics. Can be "kalman" or "custom".
        custom_func: function
            Custom function to get the kinematics.

        Returns
        -------
        kinematics: list
            List of kinematics.
        """
        marker_set_idx = [i for i, m in enumerate(self.marker_sets) if m.name == marker_set_name][0]
        return self.marker_sets[marker_set_idx].get_kinematics(model_path, method, custom_func=custom_func, **kwargs)
