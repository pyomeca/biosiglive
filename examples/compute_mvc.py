"""
This file is part of biosiglive. it is an example to see how to use biosiglive to compute the maximal voluntary
 contraction from EMG signals.
"""

from time import strftime
from biosiglive.interfaces.vicon_interface import ViconClient
from biosiglive.interfaces.pytrigno_interface import PytrignoClient
from biosiglive.interfaces.client_interface import TcpClient
from biosiglive.processing.data_processing import OfflineProcessing
from biosiglive.gui.plot import LivePlot, Plot
from time import time, sleep
import os
import scipy.io as sio
import numpy as np
from typing import Union


class ComputeMvc:
    def __init__(self,
                 stream_mode: str = "pytrigno",  # or 'server_data' or 'vicon'
                 interface_ip: str = "127.0.0.1",
                 interface_port: int = 801,  # only for vicon
                 output_file: str = None,
                 muscle_names: list = None,
                 emg_frequency: float = 2000,
                 acquisition_rate: float = 100,
                 mvc_windows: int = 2000,
                 test_with_connection: bool = True,
                 range_muscle: Union[tuple, int] = None,
                 ):
        """
        Initialize the class.

        Parameters
        ----------
        stream_mode : str
            The mode of the stream.
            'pytrigno' : use the pytrigno
            'server_data' : use the server data
            'vicon' : use the vicon
        interface_ip : str
            The ip of the interface.
        interface_port : int
            The port of the interface.
        output_file : str
            The path of the output file.
        muscle_names : list
            The list of the muscle names.
        frequency : float
            The frequency of the device.
        acquisition_rate : float
            The acquisition rate of the acquisition.
        mvc_windows : int
            size of the window to compute the mvc.
        test_with_connection : bool
            If True, the program will try to connect to the device.
        range_muscle : tuple
            The range of the muscle to compute the mvc.
        """

        self.stream_mode = stream_mode
        if muscle_names:
            self.muscle_names = muscle_names
        else:
            self.muscle_names = []
            for i in range(range_muscle[0], range_muscle[1]):
                self.muscle_names.append(f"Muscle_{i}")

        self.frequency = emg_frequency
        self.acquisition_rate = acquisition_rate
        self.mvc_windows = mvc_windows
        self.test_with_connection = test_with_connection

        current_time = strftime("%Y%m%d-%H%M")
        self.output_file = f"MVC_{current_time}.mat" if not output_file else output_file

        self.device_host = None
        self.interface_port = interface_port
        self.interface_ip = interface_ip
        self.range_muscle = range_muscle
        self.device_name = None
        self.nb_muscles = len(self.muscle_names)
        self.show_data = False
        self.plot_app, self.rplt, self.layout, self.app, self.box = None, None, None, None, None
        self.is_processing_method = False
        self.try_number = 0

        # self.bpf_lcut, self.bpf_hcut, self.lpf_lcut = None, None, None
        # self.lp_butter_order, self.bp_butter_order = None, None
        # self.ma_win = None
        self.emg_processing = None
        self.moving_average, self.low_pass, self.custom = None, None, None

        self.try_name = ""
        self.try_list = []
        self.emg_interface = None
        if not self.test_with_connection:
            pass
        else:
            if self.stream_mode == "pytrigno":
                self._init_pytrigno_emg()
            elif self.stream_mode == "vicon":
                self._init_vicon_emg()
            elif self.stream_mode == "server_data":
                self._init_server_emg()
            else:
                raise ValueError("stream_mode must be 'pytrigno', 'vicon' or 'server_data'")

    def set_processing_method(self,
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
        Set the emg processing method.

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

    def run(self, show_data: bool = False):
        """
        Run the MVC program.

        Parameters
        ----------
        show_data: bool
            If True, the data will be displayed in a plot.
        """
        self.show_data = show_data
        self.try_number = 0
        while True:
            if show_data:
                self.rplt, self.layout, self.app, self.box = self._init_live_plot(multi=True)
            nb_frame, var, duration = self._init_trial()
            c = 0
            trial_emg = self._mvc_trial(duration, nb_frame, var)
            processed_emg, raw_emg = self._process_emg(trial_emg, save_tmp=True)
            self._plot_trial(raw_emg, processed_emg)

            task = input(
                "Press 'c' to do an other MVC trial," " 'r' to do again the MVC trial or 'q' then enter to quit.\n"
            )

            while task != "c" and task != "r" and task != "q":
                print(f"Invalid entry ({task}). Please press 'c', 'r', or 'q' (in lowercase).")
                task = input(
                    "Press 'c' to do an other MVC trial," 
                    " 'r' to do again the MVC trial or 'q' then enter to quit.\n"
                )

            if task == "c":
                pass

            elif task == "r":
                self.try_number -= 1
                mat_content = sio.loadmat("_MVC_tmp.mat")
                mat_content.pop(f"{self.try_name}_processed", None)
                mat_content.pop(f"{self.try_name}_raw", None)
                sio.savemat("_MVC_tmp.mat", mat_content)
                self.try_list = self.try_list[:-1]

            elif task == "q":
                self._save_trial()
                break

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
                    print(
                        "Trial is running please press 'Ctrl+C' when trial is ended "
                        "(it will not end the program)."
                    )

                if self.test_with_connection is True:
                    data_tmp = self.emg_interface.devices[0].get_device_data(stream_now=True, get_names=True)
                else:
                    data_tmp = np.random.random((self.nb_muscles, int(self.frequency / self.acquisition_rate)))
                tic = time()

                data = data_tmp if nb_frame == 0 else np.append(data, data_tmp, axis=1)

                self._update_live_plot(data, nb_frame)
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
                    self.app.disconnect()
                    try:
                        self.app.closeAllWindows()
                    except RuntimeError:
                        pass
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
                        legend = ["Raw", "Processed"]
                    legend = legend * raw_data.shape[0]
                    x = np.linspace(0, raw_data.shape[1] / self.frequency, raw_data.shape[1])
                    print("Close the plot windows to continue.")
                    Plot().multi_plot(data,
                                      nb_column=nb_column,
                                      y_label="Activation level (v)",
                                      x_label="Time (s)",
                                      legend=legend,
                                      subplot_title=self.muscle_names,
                                      figure_name=self.try_name,
                                      x=x)
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
        emg_processed = self.emg_processing(data, self.frequency, pyomeca=self.low_pass, ma=self.moving_average)
        file_name = "_MVC_tmp.mat"
        # save tmp_file
        if save_tmp:
            if os.path.isfile(file_name):
                mat = sio.loadmat(file_name)
                mat[f"{self.try_name}_processed"] = emg_processed
                mat[f"{self.try_name}_raw"] = data
                data_to_save = mat
            else:
                data_to_save = {f"{self.try_name}_processed": emg_processed, f"{self.try_name}_raw": data}
            sio.savemat(file_name, data_to_save)
        return emg_processed, data

    def _init_live_plot(self, multi=True):
        """
        Initialize the live plot.

        Parameters
        ----------
        multi: bool
            If True, the live plot is initialized for multi-threads plot.

        Returns
        -------
        rplt: list of live plot, layout: qt layout, qt app : pyqtapp, checkbox : list of checkbox

        """
        self.plot_app = LivePlot(multi_process=multi)
        self.plot_app.add_new_plot("EMG", "curve", self.muscle_names)
        rplt, layout, app, box = self.plot_app.init_plot_window(self.plot_app.plot[0],
                                                                use_checkbox=True,
                                                                remote=True
                                                                )
        return rplt, layout, app, box

    def _update_live_plot(self, data, nb_frame):
        """
        Update the live plot.
        Parameters
        ----------
        data: numpy.ndarray
            The EMG data to plot.
        nb_frame: int
            The current frame.
        """
        if self.plot_app:
            plot_data = data if nb_frame < self.acquisition_rate else data[:, -self.mvc_windows:]
            self.plot_app.update_plot_window(self.plot_app.plot[0],
                                             plot_data,
                                             self.app,
                                             self.rplt,
                                             self.box
                                             )

    def _init_pytrigno_emg(self):
        """
        Initialize the pytrigno EMG object.
        """
        self.range_muscle = (0, 16) if not self.range_muscle else self.range_muscle
        self.nb_muscles = len(self.range_muscle)
        if self.muscle_names is None:
            self.muscle_names = []
            for i in range(self.nb_muscles):
                self.muscle_names.append(f"Muscle_{i}")

        self.emg_interface = PytrignoClient(self.interface_ip)
        self.emg_interface.add_device("EMG_trigno", self.range_muscle, type="emg", rate=self.frequency)
        # self.emg_interface.devices[-1].set

    def _init_vicon_emg(self):
        """
        Initialize the vicon EMG object.
        """
        self.emg_interface = ViconClient(self.interface_ip, self.interface_port)
        self.emg_interface.add_device(self.device_name, type="emg", rate=self.frequency)

    def _init_server_emg(self):
        """
        Initialize the server EMG object.
        """
        self.emg_interface = TcpClient(self.interface_ip, self.interface_port, type="TCP")
        self.emg_interface.add_device(self.device_name, type="emg", rate=self.frequency)

    def get_data(self):
        """
        Get the EMG data from defined emg_interface.
        """
        return self.emg_interface.devices[0].get_device_data(stream_now=True, get_names=True)

    def _save_trial(self):
        """
        Save and end the trial.
        """
        print("Concatenate data for all trials.")

        # Concatenate all trials from the tmp file.
        mat_content = sio.loadmat("_MVC_tmp.mat")
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

        mvc_list_max = np.ndarray((len(self.muscle_names), self.mvc_windows))
        mvc_trials = emg_processed
        save = True if save == 's' else False
        mvc = OfflineProcessing.compute_mvc(self.nb_muscles,
                                            mvc_trials,
                                            self.mvc_windows,
                                            mvc_list_max,
                                            '_MVC_tmp_mat',
                                            self.output_file, save)
        print(mvc)

if __name__ == "__main__":
    # number of EMG electrode
    n_electrode = 2

    # set file and directory to save
    file_name = "MVC_xxxx.mat"
    file_dir = "MVC_01_08_2021"
    device_host = "192.168.1.211"
    muscle_names = ["tri_long_1", "tri_long_2"]
    # Run MVC
    muscles_idx = (0, n_electrode - 1)
    MVC = ComputeMvc(
        stream_mode="server_data",
        output_file=file_name,
        test_with_connection=False,
        muscle_names=muscle_names,
    )
    processing_method = OfflineProcessing().process_emg()
    list_mvc = MVC.run()
    print(list_mvc)