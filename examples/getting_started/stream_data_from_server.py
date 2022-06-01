import scipy.io as sio
from biosiglive.streaming.client import Client, Message
import numpy as np
try:
    from pythonosc.udp_client import SimpleUDPClient
except ModuleNotFoundError:
    pass
from biosiglive.io.save_data import add_data_to_pickle

if __name__ == '__main__':
    # Set program variables
    read_freq = 100  # Be sure that it's the same than server read frequency
    device_host = "192.168.1.211"  # IP address of computer which run trigno device
    n_electrode = 10
    type_of_data = ["emg", "imu"]

    # load MVC data from previous trials.
    try:
        # Read data from the mvc result file (*.mat)
        list_mvc = sio.loadmat("MVC_xx_xx_xx22/MVC_xxxx.mat")["MVC_list_max"][0]
    except IOError:
        list_mvc = np.random.rand(n_electrode, 1).tolist()

    # Set file to save data
    output_file = "stream_data_xxx"
    output_dir = "test_accel"
    data_path = f"{output_dir}/{output_file}"

    # Run streaming data
    host_ip = 'localhost'
    host_port = 50000
    osc_ip = "127.0.0.1"
    osc_port = 51337
    osc_server = True
    save_data = True
    if osc_server is True:
        osc_client = SimpleUDPClient(osc_ip, osc_port)
        print("Streaming OSC activated")
    print_data = False
    count = 0
    message = Message(command=type_of_data,
                      read_frequency=read_freq,
                      nb_frame_to_get=1,
                      get_raw_data=True,
                      mvc_list=list_mvc
                      )

    while True:
        client = Client(host_ip, host_port, "TCP")
        data = client.get_data(message)
        # time.sleep(1)
        if ["emg"] in type_of_data:
            emg = np.array(data['emg'])
            raw_emg = np.array(data['raw_emg'])

        if ["imu"] in type_of_data:
            if len(np.array(data['imu']).shape) == 3:
                accel_proc = np.array(data['imu'])[:, :3, -1:]
                gyro_proc = np.array(data['imu'])[:, 3:6, -1:]
                raw_accel = np.array(data['raw_imu'])[:, :3, -1:]
                raw_gyro = np.array(data['raw_imu'])[:, 3:6, -1:]
            else:
                accel_proc = np.array(data['imu'])[:, -1:]
                gyro_proc = np.array(data['imu'])[:, -1:]
                raw_accel = np.array(data['raw_imu'])[:, :3, -1:]
                raw_gyro = np.array(data['raw_imu'])[:, 3:6, -1:]

        if print_data is True:
            if ["imu"] in type_of_data:
                print(f"Accel data :\n"
                      f"proc : {accel_proc}\n"
                      f"raw : {raw_accel}\n")
                print(f"Gyro data :\n"
                      f"proc: {gyro_proc}\n"
                      f"raw: {raw_gyro}")
            if ["emg"] in type_of_data:
                print(f'EMG data: \n'
                      f'proc: {emg}\n'
                      f'raw: {raw_emg}\n')
        if osc_server:
            if ["emg"] in type_of_data:
                emg_proc = emg[:, -1:].reshape(emg.shape[0])
                osc_client.send_message("/emg/processed/", emg.tolist())

            if ["imu"] in type_of_data:
                accel_proc = accel_proc.reshape(accel_proc.shape[0])
                gyro_proc = gyro_proc.reshape(gyro_proc.shape[0])
                osc_client.send_message("/accel/", accel_proc.tolist())
                osc_client.send_message("/gyro/", gyro_proc.tolist())

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
