from time import sleep, strftime, time
import os
from biosiglive.client import Client
from biosiglive.data_processing import process_emg
import numpy as np
from biosiglive.data_plot import init_plot_emg, update_plot_emg
import matplotlib.pyplot as plt
import scipy.io as sio
from math import ceil
try:
    from vicon_dssdk import ViconDataStream as VDS
except:
    pass
import pytrigno


class ComputeMvc:
    """
            Build the MVC trial
            ----------
            range_muscles: tuple
                Tuple of sensor to stream, note that last values is included.
            stream_mode: str
                Method to stream data. Available ones are 'pytrigno', 'viconsdk' or 'server_data'.
            output_dir: str
                Path to the output directory.
            output_file: str
                Path or name of the output file.
            muscles_names: list
                List of str(muscle names)
            device_host: str
                IP address of the trigno host (default: 'localhost')
            frequency: int
                Frequency of EMG device (default=2000)
            acquisition_rate: int
                Frequency at which the system will read sensor data.
                It will be automatically set if stream_mode is 'viconsdk'.
            trigno_dev: pytrigno.TrignoEMG
                Trigno dev if already connected else connect automatically to the host.
            vicon_stream: ViconDataStream.Client()
                The vicon client if already connected else connect automatically to the host.
            MVC_windows: int
                Size of the sliding window for EMG processing.
            server_port: int
                Port of the server while streaming with 'server_data'
            server_ip: str
                IF of the server while streaming with 'server_data'
            Returns
            -------
                class ComputeMVC
            """

    def __init__(
        self,
        range_muscles,
        stream_mode="pytrigno",  # or 'server_data' or 'vicon'
        output_dir=None,
        output_file=None,
        muscle_names=None,
        device_host=None,
        frequency=2000,
        acquisition_rate=100,
        mvc_windows=2000,
        server_port=None,
        server_ip=None,
        test_with_connection=True,
    ):
        self.test_w_connection = test_with_connection
        if self.test_w_connection is not True:
            print("[Warning] Please note that you are in 'no connection' mode for debug.")

        self.output_dir = "MVC_data" if output_dir is None else output_dir
        if os.path.isdir(self.output_dir) is not True:
            os.mkdir(self.output_dir)

        current_time = strftime("%Y%m%d-%H%M")
        self.output_file = f"MVC_{current_time}.mat" if output_file is None else output_file

        self.range_muscles = range_muscles
        self.server_port = server_port
        self.server_ip = server_ip
        self.acquisition_rate = acquisition_rate
        self.n_muscles = range_muscles[1] - range_muscles[0] + 1
        self.mvc_windows = mvc_windows
        self.try_name = ""
        self.try_list = []
        self.stream_mode = stream_mode
        self.device_host = device_host if device_host else "localhost"

        if muscle_names is None:
            self.muscle_names = []
            for i in range(self.n_muscles):
                self.muscle_names.append(f"Muscle_{i}")
        else:
            self.muscle_names = muscle_names

        self.frequency = frequency
        self.sample = int(self.frequency / self.acquisition_rate)

        if self.test_w_connection is True:
            if stream_mode == "vicon":
                # Connexion to Nexus with Vicon data stream
                print(f"Connection to ViconDataStreamSDK at : {self.device_host} ...")
                self.vicon_client = VDS.Client()
                self.vicon_client.Connect(self.device_host)
                a = self.vicon_client.GetFrame()
                while a is not True:
                    a = self.vicon_client.GetFrame()
                self.vicon_client.EnableDeviceData()
                system_rate = self.vicon_client.GetFrameRate()
                if system_rate != self.acquisition_rate:
                    print(
                        f"[Warning] Acquisition_rate is different than Vicon system rate. "
                        f"So acquisition rate is automatically set to {system_rate} Hz."
                    )
                self.acquisition_rate = system_rate

            elif stream_mode == "pytrigno":
                print("Connexion to delsys trigno...")
                self.trigno_dev = pytrigno.TrignoEMG(
                    channel_range=self.range_muscles, samples_per_read=self.sample, host=self.device_host
                )
                self.trigno_dev.start()
                print(f"Streaming data is starting on {self.n_muscles} muscles")

            elif stream_mode == "server_data":
                print("[Warning] Be sure that server is running with 'proc_EMG' flag turned to False.")

            else:
                raise RuntimeError(
                    f"Unknown stream mode ({self.stream_mode})."
                    f" Stream mode can be 'pytrigno', 'viconsdk' or 'server_data."
                )
        else:
            self.emg_exp = np.random.rand(self.n_muscles, 1500)
            # self.emg_exp = sio.loadmat("EMG_test.mat")["EMG"][:, :1500]

    def _process_mvc(self, data, save_final_data=False, save_tmp=False, return_list=False):
        """
                Process MVC signal
                ----------
                data: np.array()
                    EMG data with same size than the EMG_windows
                save_final_data: bool
                    True to save the data at the end of the trials in a .mat file
                save_tmp: bool
                    True to save a temporary file.
                return_list
                    True to return the list of MVC, else it return MVC processed.
                Returns
                -------
                list of mvc or MVC processed
                """
        mvc_trials = []
        mvc_processed = []
        windows = self.mvc_windows
        mvc_list_max = np.ndarray((len(self.muscle_names), windows))
        if save_final_data is not True and return_list is not True:
            mvc_processed = process_emg(data, self.frequency, ma_win=200, pyomeca=False, ma=True)
        else:
            mvc_trials = data

        file_name = "_MVC_tmp.mat"
        # save tmp_file
        if save_tmp:
            if os.path.isfile(file_name):
                mat = sio.loadmat(file_name)
                mat[f"{self.try_name}_processed"] = mvc_processed
                mat[f"{self.try_name}_raw"] = data
                data_to_save = mat
            else:
                data_to_save = {f"{self.try_name}_processed": mvc_processed, f"{self.try_name}_raw": data}
            sio.savemat(file_name, data_to_save)
            return mvc_processed

        if save_final_data or return_list:
            for i in range(self.n_muscles):
                mvc_temp = -np.sort(-mvc_trials, axis=1)
                if i == 0:
                    mvc_list_max = mvc_temp[:, :windows]
                else:
                    mvc_list_max = np.concatenate((mvc_list_max, mvc_temp[:, :windows]), axis=1)
            mvc_list_max = -np.sort(-mvc_list_max, axis=1)[:, :windows]
            mvc_list_max = np.mean(mvc_list_max, axis=1)
            mat_content = sio.loadmat(file_name)
            mat_content["MVC_list_max"] = mvc_list_max

            if save_final_data:
                sio.savemat(f"{self.output_dir}/{self.output_file}", mat_content)
            os.remove(file_name)
            return mvc_list_max

    def _plot_mvc(self, raw_data, proc_data, command, col=4):
        """
                Plot data
                ----------
                raw_data: np.array()
                    raw data to plot of size (nb_muscles, nb_frames)
                proc_data : np.array()
                    processed data to plot of size (nb_muscles, nb_frames)
                command: str()
                    command to know which data to plot
                col: int
                    number of columns wanted in the plot.

                Returns
                -------
                """
        data = proc_data if command == "p" else raw_data
        plt.figure(self.try_name)
        size_police = 12
        col = self.n_muscles if self.n_muscles <= 4 else col
        line = ceil(self.n_muscles / col)
        for i in range(self.n_muscles):
            plt.subplot(line, col, i + 1)
            if i % 4 == 0:
                plt.ylabel("Activation level", fontsize=size_police)
            plt.plot(data[i, :], label="raw_data")
            if command == "b":
                plt.plot(proc_data[i, :], label="proc_data")
                plt.legend()
            plt.title(self.muscle_names[i], fontsize=size_police)
        plt.show()

    def run(self, device_name=None, show_data=False, return_dev=False):
        """
                Run a MVC session.
                ----------
                    device_name: str
                        Name of the device in Nexus software. Only for 'viconsdk' or 'server_data'.
                    return_dev: bool
                        True to return the instance of streaming system to use in an other code.
                Returns
                -------
                    List of MVC and if wanted instance of streaming system.
                """
        device_info = []
        if self.test_w_connection is True:
            if self.stream_mode == "vicon":
                self.vicon_client.GetFrame()
                device_name = device_name if device_name else self.vicon_client.GetDeviceNames()[2][0]
                device_info = self.vicon_client.GetDeviceOutputDetails(device_name)
                # a = self.vicon_client.GetDeviceNames()
                # while len(a) == 0:
                #     self.vicon_client.GetFrame()
                #     a = self.vicon_client.GetDeviceNames()

        try_number = 0
        while True:
            if self.stream_mode == "pytrigno" or "server_data":
                data_tmp, data = [], []
            else:
                data_tmp, data = np.ndarray((len(device_info), self.sample)), []

            try_name = input("Please enter a name of your trial (string) then press enter or press enter.\n")
            if try_name == "":
                self.try_name = f"MVC_{try_number}"
            else:
                self.try_name = f"{try_name}"
            try_number += 1
            self.try_list.append(self.try_name)
            t = input(
                f"Ready to start trial: {self.try_name}, with muscles :{self.muscle_names}. "
                f"Press enter to begin your MVC. or enter a number of seconds"
            )
            nb_frame = 0
            try:
                float(t)
                # if isinstance(t, (int, float)):
                iter = float(t) * self.acquisition_rate
                var = iter
                duration = True
            except:
                var = -1
                duration = False
            c = 0
            if show_data is True:
                p, win_emg, app, box = init_plot_emg(self.n_muscles, self.muscle_names)

            while True:
                # os.system('cls' if os.name == 'nt' else 'clear')
                try:
                    if nb_frame == 0:
                        print(
                            "Trial is running please press 'Ctrl+C' when trial is terminated "
                            "(it will not end the program)."
                        )
                    if self.test_w_connection is True:
                        if self.stream_mode == "pytrigno":
                            # self.trigno_dev.reset()
                            # self.trigno_dev.start()
                            data_tmp = self.trigno_dev.read()
                            tic = time()
                            # assert data_tmp.shape == (self.n_muscles, self.sample)  # Reshape trigno data

                        elif self.stream_mode == "vicon":
                            self.vicon_client.GetFrame()
                            tic = time()
                            mucles_idx = 0
                            for output_name, EMG_name, unit in device_info:
                                data_tmp[mucles_idx, :], occluded = self.vicon_client.GetDeviceOutputValues(
                                    device_name, output_name, EMG_name
                                )
                                mucles_idx += 1

                        elif self.stream_mode == "server_data":
                            client = Client(self.server_ip, self.server_port)
                            data_tmp = client.get_data(
                                ["emg"],
                                nb_frame_of_interest=self.sample,
                                nb_of_data_to_export=self.sample,
                                read_frequency=self.acquisition_rate,
                                emg_wind=2000,
                                get_names=False,
                                raw=True,
                                norm_emg=False,
                            )
                            data_tmp = np.array(data_tmp["raw_emg"])
                            tic = time()

                    else:
                        if c < self.emg_exp.shape[1]:
                            data_tmp = self.emg_exp[: self.n_muscles, c : c + self.sample]
                            c += self.sample
                        else:
                            c = 0
                        tic = time()
                    if nb_frame == 0:
                        data = data_tmp
                    else:
                        data = np.append(data, data_tmp, axis=1)  # append data with new samples data

                    if show_data is True:
                        if nb_frame < self.acquisition_rate:
                            update_plot_emg(data, p, app, box)
                        else:
                            update_plot_emg(data[:, -self.mvc_windows :], p, app, box)

                    nb_frame += 1
                    time_to_sleep = (1 / self.acquisition_rate) - (time() - tic)
                    if time_to_sleep > 0:
                        sleep(time_to_sleep)
                    else:
                        print(f"Delay of {abs(time_to_sleep)}.")
                    if duration:
                        if nb_frame == var:
                            break
                except KeyboardInterrupt:
                    if show_data is True:
                        app.disconnect()
                        try:
                            app.closeAllWindows()
                        except:
                            pass
                    break

            mvc_processed = self._process_mvc(data, save_tmp=True)
            n_p = 0
            plot_comm = "y"
            print(f"Trial {try_name} terminated. ")
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
                        print("Close the plot windows to continue.")
                        self._plot_mvc(data, mvc_processed, plot, col=4)
                    else:
                        pass
                    n_p += 1

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
                try_number -= 1
                mat_content = sio.loadmat("_MVC_tmp.mat")
                mat_content.pop(f"{self.try_name}_processed", None)
                mat_content.pop(f"{self.try_name}_raw", None)
                sio.savemat("_MVC_tmp.mat", mat_content)
                self.try_list = self.try_list[:-1]

            elif task == "q":
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
                if save == "s":
                    list_mvc = self._process_mvc(data_final, save_final_data=True, return_list=True)
                    break
                else:
                    list_mvc = self._process_mvc(data_final, save_final_data=False, return_list=True)
                    break

        return list_mvc
