import scipy.io as sio
from biosiglive.client import Client
import numpy as np
import os
from pythonosc.udp_client import SimpleUDPClient
from biosiglive.data_processing import add_data_to_pickle

if __name__ == '__main__':
    # Set program variables
    read_freq = 100  # Be sure that it's the same than server read frequency
    device_host = "192.168.1.211"  # IP address of computer which run trigno device
    n_electrode = 10
    type_of_data = ["emg", "imu"]

    # load MVC data from previous trials.
    list_mvc = mvc_list = np.random.rand(n_electrode, 1)

    # Set file to save data
    output_file = "stream_data_xxx"
    output_dir = "test_accel"
    data_path = f"{output_dir}/{output_file}"

    # Run streaming data
    host_ip = 'localhost'
    host_port = 50000
    print_data = False
    save_data = True
    count = 0
    while True:
        client = Client(host_ip, host_port, "TCP")
        data = client.get_data(data=type_of_data,
                               nb_frame_of_interest=read_freq,
                               read_frequency=read_freq,
                               raw=True,
                               norm_emg=True,
                               mvc_list=list_mvc
                               )
        EMG = np.array(data['emg'])
        raw_emg = np.array(data['raw_emg'])

        if len(np.array(data['imu']).shape) == 3:
            accel_proc = np.array(data['imu'])[:, :3, :]
            gyro_proc = np.array(data['imu'])[:, 3:6, :]
            raw_accel = np.array(data['raw_imu'])[:, :3, :]
            raw_gyro = np.array(data['raw_imu'])[:, 3:6, :]
        else:
            accel_proc = np.array(data['imu'])[:, :]
            gyro_proc = np.array(data['imu'])[:, :]
            raw_accel = np.array(data['raw_imu'])[:, :3, :]
            raw_gyro = np.array(data['raw_imu'])[:, 3:6, :]

        if print_data is True:
            print(f"Accel data :\n"
                  f"proc : {accel_proc}\n"
                  f"raw : {raw_accel}\n")
            print(f"Gyro data :\n"
                  f"proc: {gyro_proc}\n"
                  f"raw: {raw_gyro}")
            print(f'EMG data: \n'
                  f'proc: {EMG}\n'
                  f'raw: {raw_emg}\n')

        if save_data is True:
            if count == 0:
                print("Save data starting.")
                count += 1
            for key in data.keys():
                if key == 'imu':
                    if len(np.array(data['imu']).shape) == 3:
                        data[key] = np.array(data[key])
                        data['accel_proc'] = data[key][:n_electrode, :3, :]
                        data['gyro_proc'] = data[key][n_electrode:, 3:6, :]
                    else:
                        data[key] = np.array(data[key])
                        data['accel_proc'] = data[key][:n_electrode, :]
                        data['gyro_proc'] = data[key][n_electrode:, :]
                else:
                    data[key] = np.array(data[key])
            add_data_to_pickle(data, data_path)
