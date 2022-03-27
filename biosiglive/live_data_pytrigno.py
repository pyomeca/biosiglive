try:
    import pytrigno
except ModuleNotFoundError:
    pass
from biosiglive.data_plot import init_plot_emg, update_plot_emg
from biosiglive.data_processing import process_emg_rt, process_imu, add_data_to_pickle
from time import sleep, time, strftime
import os
import numpy as np
import scipy.io as sio

try:
    from pythonosc.udp_client import SimpleUDPClient

    osc_package = True
except ModuleNotFoundError:
    osc_package = False


def run(
    muscles_range,
    get_emg=True,
    get_accel=True,
    get_gyro=True,
    read_freq=100,
    mvc_list=(),
    norm_min_bound_accel=None,
    norm_max_bound_accel=None,
    norm_min_bound_gyro=None,
    norm_max_bound_gyro=None,
    host_ip=None,
    emg_freq=2000,
    IM_freq=148.1,
    emg_windows=2000,
    IM_windows=100,
    accept_delay=0.005,
    save_data=True,
    output_dir=None,
    output_file=None,
    show_data=None,
    print_data=False,
    server="pytrigno",
    norm_emg=False,
    muscle_names=(),
    test_with_connection=True,
    OSC_stream=False,
    OSC_ip="127.0.0.1",
    OSC_port=51337,
):
    """
        Run streaming of delsys sensors data with real time processing and plotting.
        ----------
        muscles_range: tuple
            list of sensor to stream, note that last values is included.
        get_emg: bool
            True to stream emg data
        get_accel: bool
            True to stream accelerometer data
        get_gyro: bool
            True to stream gyroscope data
        read_freq: int
            frequency at which the system will read sensor data.
        MVC_list: list
            list of MVC value length must be the same than sensor number.
        host_ip: str
            IP adress of the device which run the trigno software default is 'localhost'.
        emg_freq: float
            frequency of emg data.
        IM_freq: float
            frequency of inertial measurement data.
        emg_windows: float
            size of the sliding window for emg processing.
        IM_windows: float
            size of the sliding window for IM processing.
        accept_delay: float
            acceptable delay between real time and data streaming.
        save_data: bool
            True for save data during streaming, this can impact the realtime streaming.
        output_dir: str
            name of output directory.
        output_file: str
            name of output file.
        show_data: list
            list of name of data to plot. Can be: 'emg' or 'raw_emg' (gyro and accel not implemented yet).
        print_data: bool
            True to print data in the console
        server: str
            method to stream data. Can be 'pytrigno'.
        norm_emg: bool
            True for normalize emg in real time. Note that you need a MVC list.
        muscle_names: list
            list of muscle names. Length must be the same than the number of delsys sensors.
        OSC_stream: bool
            Stream OSC (open sound control) value to destination
        OSC_port: int
            OSC output port (must be over 1024 and under 65000), default : 51337 
        OSC_ip: str
            OSC output ip address, default : 127.0.0.1        
        Returns
            -------
     """
    if len(mvc_list) != 0:
        norm_emg = True
    if test_with_connection is not True:
        print("[WARNING] Please note that you are in 'no connection' mode for debug.")

    if get_emg is False and get_accel is False and get_gyro is False:
        raise RuntimeError("Please define at least one data to read (emg/gyro/accel).")
    if get_gyro is True:
        print("[WARNING] Please note that only avanti sensor have gyroscope data available.")

    if show_data:
        for data in show_data:
            if data == "accelerometer" or data == "accel":
                raise RuntimeError("Plot accelerometer data not implemented yet.")
            elif data == "gyroscope" or data == "gyro":
                raise RuntimeError("Plot gyroscope data not implemented yet.")

    if server == "vicon" and get_accel is True or server == "vicon" and get_gyro is True:
        raise RuntimeError("Read IM data with vicon not implemented yet")

    current_time = strftime("%Y%m%d-%H%M")
    output_file = output_file if output_file else f"trigno_streaming_{current_time}"

    output_dir = output_dir if output_dir else "live_data"

    if os.path.isdir(output_dir) is not True:
        os.mkdir(output_dir)

    if os.path.isfile(f"{output_dir}/{output_file}"):
        os.remove(f"{output_dir}/{output_file}")

    data_path = f"{output_dir}/{output_file}"

    if get_accel is not True and get_emg is not True and get_gyro is not True:
        raise RuntimeError("Please define at least one data to read.")

    OSC_client = []
    dev_emg = []
    dev_IM = []
    if isinstance(muscles_range, tuple) is not True:
        raise RuntimeError("muscles_range must be a tuple.")
    n_muscles = muscles_range[1] - muscles_range[0] + 1

    emg_sample = int(emg_freq / read_freq)
    IM_sample = int(IM_freq / read_freq)
    IM_range = (muscles_range[0], muscles_range[0] + (n_muscles * 9))

    host_ip = "localhost" if None else host_ip
    if test_with_connection is True:
        if server == "pytrigno":
            if get_emg is True:
                dev_emg = pytrigno.TrignoEMG(channel_range=muscles_range, samples_per_read=emg_sample, host=host_ip)
                dev_emg.start()
            if get_accel is True or get_gyro is True:
                dev_IM = pytrigno.TrignoIM(channel_range=IM_range, samples_per_read=IM_sample, host=host_ip)
                dev_IM.start()

    data_emg_tmp = []
    data_IM_tmp, accel_proc, raw_accel, gyro_proc, raw_gyro = [], [], [], [], []
    raw_emg, emg_proc, emg_to_plot = [], [], []

    if show_data:
        p, win_emg, app, box = init_plot_emg(n_muscles, muscle_names)

    if len(muscle_names) == 0:
        muscle_names = []
        for i in range(n_muscles):
            muscle_names.append("muscle_" + f"{i}")

    print("Streaming data.....")
    ma_win = 200
    if test_with_connection is not True:
        data = sio.loadmat("test_imu.mat")
        IM_exp = np.concatenate((data["raw_accel"][:, :, :6000], data["raw_gyro"][:, :, :6000]), axis=1)
        emg_exp = sio.loadmat("emg_test.mat")["emg"][:, :1500]
    c = 0
    d = 0
    initial_time = time()

    if OSC_stream is True and osc_package is not True:
        raise RuntimeError(
            "OSC package need to be installed to use OSC stream." " Please turn off the flag or install pythonosc."
        )

    elif OSC_stream is True and osc_package:
        OSC_client = SimpleUDPClient(OSC_ip, OSC_port)
        print("Streaming OSC activated")

    while True:
        if test_with_connection:
            if server == "pytrigno":
                if get_emg is True:
                    data_emg_tmp = dev_emg.read()

                if get_accel is True or get_gyro is True:
                    data_IM_tmp = dev_IM.read()
                    data_IM_tmp = data_IM_tmp.reshape(n_muscles, 9, IM_sample)
            else:
                raise RuntimeError(f"Server '{server}' not valid, please use 'pytrigno' server.")
        else:
            if get_emg is True:
                if c < emg_exp.shape[1]:
                    data_emg_tmp = emg_exp[:n_muscles, c : c + emg_sample]
                    c += emg_sample
                else:
                    c = 0
            if get_accel is True:
                if d < IM_exp.shape[2]:
                    data_IM_tmp = IM_exp[:n_muscles, :, d : d + IM_sample]
                    d += IM_sample
                else:
                    d = 0
                # data_IM_tmp = np.random.random((n_muscles, 9, IM_sample))
        tic = time()

        if get_emg is True:
            raw_emg, emg_proc = process_emg_rt(
                raw_emg,
                emg_proc,
                data_emg_tmp,
                mvc_list=mvc_list,
                ma_win=ma_win,
                emg_win=emg_windows,
                emg_freq=emg_freq,
                norm_emg=norm_emg,
                lpf=False,
            )

            if show_data:
                for data in show_data:
                    if data == "raw_emg":
                        if raw_emg.shape[1] < emg_windows:
                            emg_to_plot = np.append(
                                np.zeros((raw_emg.shape[0], emg_windows - raw_emg.shape[1])), raw_emg, axis=1
                            )
                        else:
                            emg_to_plot = raw_emg
                    elif data == "emg":
                        if emg_proc.shape[1] < read_freq:
                            emg_to_plot = np.append(
                                np.zeros((emg_proc.shape[0], read_freq - emg_proc.shape[1])), emg_proc, axis=1
                            )
                        else:
                            emg_to_plot = emg_proc

            # print emg data
            if print_data is True:
                print(f"emg processed data :\n {emg_proc[:, -1:]}")
            if OSC_stream is True:
                OSC_client.send_message("/emg/processed/", emg_proc[:, -1:])

        if get_accel is True or get_gyro is True:
            accel_tmp = data_IM_tmp[:, :3, :]
            gyro_tmp = data_IM_tmp[:, 3:6, :]
            raw_accel, accel_proc = process_imu(
                accel_proc,
                raw_accel,
                accel_tmp,
                IM_windows,
                IM_sample,
                ma_win=30,
                norm_min_bound=norm_min_bound_accel,
                norm_max_bound=norm_max_bound_accel,
            )

            raw_gyro, gyro_proc = process_imu(
                gyro_proc,
                raw_gyro,
                gyro_tmp,
                IM_windows,
                IM_sample,
                ma_win=30,
                norm_min_bound=norm_min_bound_gyro,
                norm_max_bound=norm_max_bound_gyro,
            )

            # Print IM data
            if print_data is True:
                print(f"Accel data :\n {accel_proc[:, :, -1:]}")
                print(f"Gyro data :\n {gyro_proc[:, :, -1:]}")

            if OSC_stream is True:
                if get_accel is True:
                    OSC_client.send_message("/accel/", accel_proc[:, :, -1:])
                if get_gyro is True:
                    OSC_client.send_message("/gyro/", gyro_proc[:, :, -1:])

        # Save data
        if save_data is True:
            data_to_save = {
                "Time": time() - initial_time,
                "emg_freq": emg_freq,
                "IM_freq": IM_freq,
                "read_freq": read_freq,
            }
            if get_emg is True:
                data_to_save["emg_proc"] = emg_proc[:, -1:]
                data_to_save["raw_emg"] = data_emg_tmp
            if get_accel is True:
                data_to_save["accel_proc"] = accel_proc[:, :, -1:]
                data_to_save["raw_accel"] = data_IM_tmp[:, 0:3, :]
            if get_gyro is True:
                data_to_save["gyro_proc"] = gyro_proc[:, :, -1:]
                data_to_save["raw_gyro"] = data_IM_tmp[:, 3:6, :]

            add_data_to_pickle(data_to_save, data_path)

        # Plot data real time
        if show_data:
            for data in show_data:
                if data == "raw_emg" or data == "emg":
                    update_plot_emg(emg_to_plot, p, app, box)

                else:
                    raise RuntimeError(
                        f"{data} is unknown. Please define a valid type of data to plot ('emg' or raw_emg')."
                    )

        t = time() - tic
        time_to_sleep = 1 / read_freq - t
        if time_to_sleep > 0:
            sleep(time_to_sleep)
        elif float(abs(time_to_sleep)) < float(accept_delay):
            pass
        else:
            print(
                f"[Warning] Processing need to much time and delay ({abs(time_to_sleep)}) exceeds "
                f"the threshold ({accept_delay}). Try to reduce the read frequency or emg frequency."
            )
