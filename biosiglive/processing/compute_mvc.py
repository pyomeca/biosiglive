"""
This file is part of biosiglive. it is an example to see how to use biosiglive to compute the maximal voluntary
 contraction from EMG signals.
"""

from time import strftime
from ..interfaces.vicon_interface import ViconClient
from ..interfaces.pytrigno_interface import PytrignoClient
from ..interfaces.tcp_interface import TcpClient
from ..interfaces.generic_interface import GenericInterface
from .data_processing import OfflineProcessing
from ..gui.plot import LivePlot, OfflinePlot
from ..enums import InterfaceType
from ..file_io.save_and_load import save, load
from pathlib import Path
from time import time, sleep
import os
import numpy as np
from typing import Union


class ComputeMvc:
    def __init__(
        self,
        interface_type: Union[str, InterfaceType] = InterfaceType.PytrignoClient,
        interface_ip: str = "127.0.0.1",
        interface_port: int = 801,  # only for vicon
        output_file: str = None,
        muscle_names: list = None,
        emg_frequency: float = 2000,
        acquisition_rate: int = 100,
        mvc_windows: int = 2000,
        range_muscle: Union[tuple, int] = None,
        custom_interface: GenericInterface = None,
    ):
        """
        Initialize the class.

        Parameters
        ----------
        interface_type : Union[str, InterfaceType]
            The mode of the stream (describe in the InterfaceType enum).
        interface_ip : str
            The ip of the interface.
        interface_port : int
            The port of the interface. Only needed for the Vicon interface.
        output_file : str
            The path of the output file.
        muscle_names : list
            The list of the muscle names.
        emg_frequency : float
            The frequency of the device.
        acquisition_rate : float
            The acquisition rate of the acquisition.
        mvc_windows : int
            size of the window to compute the mvc.
        range_muscle : tuple
            The range of the muscle to compute the mvc.
        custom_interface: classmethod
            The custom interface to use
        """
        if Path(output_file).suffix != ".bio":
            if Path(output_file).suffix == "":
                output_file += ".bio"
            else:
                raise ValueError("The file must be a .bio file.")

        if isinstance(interface_type, str):
            if interface_type not in [t.value for t in InterfaceType]:
                raise ValueError("The type of the interface is not valid.")
            interface_type = InterfaceType(interface_type)
        self.interface_type = interface_type
        self.range_muscle = range_muscle if range_muscle else (0, 16)
        self.nb_muscles = self.range_muscle[1] - self.range_muscle[0]
        if muscle_names:
            self.muscle_names = muscle_names
        else:
            self.muscle_names = []
            for i in range(self.range_muscle[0], self.range_muscle[1]):
                self.muscle_names.append(f"Muscle_{i}")
        if self.nb_muscles != len(self.muscle_names):
            raise RuntimeError("number of muscle must have the same length than names.")
        self.frequency = emg_frequency
        self.acquisition_rate = acquisition_rate
        self.mvc_windows = mvc_windows

        current_time = strftime("%Y%m%d-%H%M")
        self.output_file = f"MVC_{current_time}.bio" if not output_file else output_file

        self.device_host = None
        self.interface_port = interface_port
        self.interface_ip = interface_ip
        self.device_name = None
        self.show_data = False
        self.is_processing_method = False
        self.try_number = 0
        self.emg_plot = None

        self.emg_processing = None
        self.moving_average, self.low_pass, self.custom = None, None, None

        self.try_name = ""
        self.try_list = []
        self.emg_interface = None

        if self.interface_type == InterfaceType.PytrignoClient:
            self.emg_interface = PytrignoClient(system_rate=self.acquisition_rate, ip=self.interface_ip)
        elif self.interface_type == InterfaceType.ViconClient:
            self.emg_interface = ViconClient(
                system_rate=self.acquisition_rate, ip=self.interface_ip, port=self.interface_port
            )
        elif self.interface_type == InterfaceType.TcpClient:
            self.emg_interface = TcpClient(
                read_frequency=self.acquisition_rate, ip=self.interface_ip, port=self.interface_port
            )
        elif self.interface_type == InterfaceType.Custom:
            self.emg_interface = custom_interface

        if self.interface_type != InterfaceType.Custom:
            self.emg_interface.add_device(nb_channels=self.nb_muscles, name="EMG", device_range=self.range_muscle)
        else:
            if len(self.emg_interface.devices) == 0:
                raise RuntimeError("Please add a device in the custom interface.")

    def set_processing_method(
        self,
        moving_average: bool = True,
        low_pass: bool = False,
        custom: bool = False,
        custom_function: callable = None,
        bandpass_frequency: tuple = (10, 425),
        lowpass_frequency: float = 5,
        lowpass_order: int = 4,
        butterworth_order: int = 4,
        ma_window: int = 200,
    ):
        """
        Set the emg processing method. This method allow to customize the processing of the emg signal,
        so it need to be called before the start of the acquisition.

        Parameters
        ----------
        moving_average : bool
            If True, the emg data will be processed with a moving average.
        low_pass : bool
            If True, the emg data will be processed with a low pass filter.
        custom : bool
            If True, the emg data will be processed with a custom function.
        custom_function : callable
            The custom function. Input : raw data, device frequency Output : processed data.
        bandpass_frequency : tuple
            The frequency of the bandpass filter.
        lowpass_frequency : float
            The frequency of the low pass filter.
        lowpass_order : int
            The order of the low pass filter.
        butterworth_order : int
            The order of the butterworth filter.
        ma_window : int
            The size of the moving average window.
        """

        self.moving_average = moving_average
        self.low_pass = low_pass
        self.custom = custom
        if [moving_average, custom, low_pass].count(True) > 1:
            raise ValueError("Only one processing method can be selected")
        if custom and not custom_function:
            raise ValueError("custom_function must be defined")
        if custom:
            self.emg_processing = custom_function
        else:
            emg_processing = OfflineProcessing()
            emg_processing.bpf_lcut = bandpass_frequency[0]
            emg_processing.bpf_hcut = bandpass_frequency[1]
            emg_processing.lpf_lcut = lowpass_frequency
            emg_processing.lp_butter_order = lowpass_order
            emg_processing.bp_butter_order = butterworth_order
            emg_processing.ma_win = ma_window
            self.emg_processing = emg_processing.process_emg
        self.is_processing_method = True

    def run(self, show_data: bool = False) -> list:
        """
        Run the MVC program.

        Parameters
        ----------
        show_data: bool
            If True, the data will be displayed in live in a separate plot.

        Returns
        -------
        list
            The list of the MVC value for each muscle.
        """
        self.show_data = show_data
        self.try_number = 0
        while True:
            if show_data:
                self.emg_plot = LivePlot(nb_subplots=self.nb_muscles, channel_names=self.muscle_names)
                self.emg_plot.init(plot_windows=int(self.frequency * 2))
            nb_frame, var, duration = self._init_trial()
            trial_emg = self._mvc_trial(duration, nb_frame, var)
            processed_emg, raw_emg = self._process_emg(trial_emg, save_tmp=True)
            self._plot_trial(raw_emg, processed_emg)

            task = input(
                "Press 'c' to do an other MVC trial," " 'r' to do again the MVC trial or 'q' then enter to quit.\n"
            )

            while task != "c" and task != "r" and task != "q":
                print(f"Invalid entry ({task}). Please press 'c', 'r', or 'q' (in lowercase).")
                task = input(
                    "Press 'c' to do an other MVC trial," " 'r' to do again the MVC trial or 'q' then enter to quit.\n"
                )

            if task == "c":
                pass

            elif task == "r":
                self.try_number -= 1
                mat_content = load("_MVC_tmp.bio")
                mat_content.pop(f"{self.try_name}_processed", None)
                mat_content.pop(f"{self.try_name}_raw", None)
                save(mat_content, "_MVC_tmp.bio")
                self.try_list = self.try_list[:-1]

            elif task == "q":
                mvc_list = self._save_trial()
                break
        return mvc_list

    def _init_trial(self):
        """
        Initialize the trial.

        Returns
        -------
        nb_frame : int
            The number of frames of the trial.
        var : float
            The current iteration.
        duration : float
            The duration of the trial.
        """

        try_name = input("Please enter a name of your trial (string) then press enter or press enter.\n")
        while try_name in self.try_list:
            try_name = input("This name is already used. Please chose and other name.\n")

        if try_name == "":
            self.try_name = f"MVC_{self.try_number}"
        else:
            self.try_name = f"{try_name}"
        self.try_number += 1

        self.try_list.append(self.try_name)
        t = input(
            f"Ready to start trial: {self.try_name}, with muscles :{self.muscle_names}. "
            f"Press enter to begin your MVC. or enter a number of seconds."
        )
        nb_frame = 0
        try:
            float(t)
            iter = float(t) * self.acquisition_rate
            var = iter
            duration = True
        except ValueError:
            var = -1
            duration = False
        return nb_frame, var, duration

    def _mvc_trial(self, duration: float, nb_frame: int, var: float):
        """
        Run the MVC trial.

        Parameters
        ----------
        duration : float
            The duration of the trial.
        nb_frame : int
            The number of frames of the trial.
        var : float
            The current iteration.

        Returns
        -------
        trial_emg : numpy.ndarray
            The EMG data of the trial.
        """
        data = None
        while True:
            try:
                if nb_frame == 0:
                    print("Trial is running please press 'Ctrl+C' when trial is ended (it will not end the program).")

                data_tmp = self.emg_interface.get_device_data()
                tic = time()

                data = data_tmp if nb_frame == 0 else np.append(data, data_tmp, axis=1)
                if self.show_data:
                    self.emg_plot.update(data_tmp)
                nb_frame += 1

                time_to_sleep = (1 / self.acquisition_rate) - (time() - tic)

                if time_to_sleep > 0:
                    sleep(time_to_sleep)
                else:
                    print(f"Delay of {abs(time_to_sleep)}.")

                if duration:
                    if nb_frame == var:
                        return data

            except KeyboardInterrupt:
                if self.show_data is True:
                    self.emg_plot.disconnect()
                return data

    def _plot_trial(self, raw_data: np.ndarray = None, processed_data: np.ndarray = None):
        """
        Plot the trial.

        Parameters
        ----------
        raw_data : numpy.ndarray
            The raw EMG data of the trial.
        processed_data : numpy.ndarray
                The processed EMG data of the trial.
        """
        data = raw_data
        legend = ["Raw"]
        nb_column = 4 if raw_data.shape[0] > 4 else raw_data.shape[0]
        n_p = 0
        plot_comm = "y"
        print(f"Trial {self.try_name} terminated. ")
        while plot_comm != "n":
            if n_p != 0:
                plot_comm = input(f"Would you like to plot again ? 'y'/'n'")

            if plot_comm == "y":
                plot = input(
                    f"Press 'pr' to plot your raw trial,"
                    f" 'p' to plot your processed trial, 'b' to plot both or 'c' to continue,"
                    f" then press enter."
                )
                while plot != "p" and plot != "pr" and plot != "c" and plot != "b":
                    print(f"Invalid entry ({plot}). Please press 'p', 'pr', 'b',  or 'c' (in lowercase).")
                    plot = input(
                        f"Press 'pr' to plot your raw trial,"
                        f"'p' to plot your processed trial or 'c' to continue then press enter."
                    )

                if plot != "c":
                    if plot == "p":
                        data = processed_data
                        legend = ["Processed"]
                    elif plot == "b":
                        data = [raw_data, processed_data]
                        legend = ["Raw data", "Processed"]
                    legend = legend * raw_data.shape[0]
                    x = np.linspace(0, raw_data.shape[1] / self.frequency, raw_data.shape[1])
                    print("Close the plot windows to continue.")
                    OfflinePlot().multi_plot(
                        data,
                        nb_column=nb_column,
                        y_label="Activation level (v)",
                        x_label="Time (s)",
                        legend=legend,
                        subplot_title=self.muscle_names,
                        figure_name=self.try_name,
                        x=x,
                    )
                else:
                    pass
                n_p += 1

    def _process_emg(self, data, save_tmp=True):
        """
        Process the EMG data.

        Parameters
        ----------
        data : numpy.ndarray
            The raw EMG data of the trial.
        save_tmp : bool
            If True, the processed data is saved in a temporary file.

        Returns
        -------
        numpy.ndarray
            The processed EMG data of the trial.
        """
        if not self.is_processing_method:
            self.set_processing_method()
        emg_processed = self.emg_processing(
            data, data_rate=self.frequency, moving_average=self.moving_average, low_pass_filter=self.low_pass
        )
        file_name = "_MVC_tmp.bio"
        # save tmp_file
        if save_tmp:
            if os.path.isfile(file_name):
                mat = load(file_name)
                mat[f"{self.try_name}_processed"] = emg_processed
                mat[f"{self.try_name}_raw"] = data
                data_to_save = mat
            else:
                data_to_save = {f"{self.try_name}_processed": emg_processed, f"{self.try_name}_raw": data}
            save(data_to_save, file_name)
        return emg_processed, data

    def _save_trial(self) -> list:
        """
        Save and end the trial.

        Returns
        -------
        list
            The list of the mvc for each muscle.
        """

        print("Concatenate data for all trials.")
        mat_content = load("_MVC_tmp.bio")
        data_final = []
        for i in range(len(self.try_list)):
            if i == 0:
                data_final = mat_content[f"{self.try_list[i]}_processed"]
            else:
                data_final_tmp = mat_content[f"{self.try_list[i]}_processed"]
                data_final = np.append(data_final, data_final_tmp, axis=1)

        save = input("Press 's' to save your data, other key to just return a list of MVC.\n")
        if save != "s":
            save = input("Data will not be saved. " "If you want to save press 's', if not, press enter.\n")

        print("Please wait during data processing (it could take some time)...")
        emg_processed, data_raw = self._process_emg(data_final, save_tmp=False)
        mvc_trials = emg_processed
        save = True if save == "s" else False
        mvc = OfflineProcessing.compute_mvc(
            self.nb_muscles, mvc_trials, self.mvc_windows, "_MVC_tmp.bio", self.output_file, save
        )
        return mvc
