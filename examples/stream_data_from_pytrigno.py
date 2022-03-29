from biosiglive.live_data_pytrigno import run
import scipy.io as sio

if __name__ == "__main__":
    # Set program variables
    read_freq = 100
    n_electrode = 4
    show_data = ["raw_emg"]  # can be ["emg"] to show process EMG
    device_host = "192.168.1.211"  # IP address of computer which run trigno device

    # load MVC data from previous trials.
    file_name = "MVC_xxxx.mat"
    file_dir = "MVC_01_08_2021"
    list_mvc = sio.loadmat(f"{file_dir}/{file_name}")["MVC_list_max"]
    list_mvc = list_mvc[:, :n_electrode].T
    # Set file to save data
    output_file = "stream_data_xxx"
    output_dir = "test_accel"

    # Run streaming data
    muscles_idx = (0, n_electrode - 1)
    run(
        muscles_idx,
        output_file=output_file,
        output_dir=output_dir,
        read_freq=read_freq,
        host_ip=device_host,
        norm_emg=False,
        norm_min_bound_accel=-2,  # Can be positive or negative value
        norm_max_bound_accel=2,
        norm_min_bound_gyro=-500,  # Can be positive or negative value
        norm_max_bound_gyro=500,
        mvc_list=list_mvc,
        show_data=show_data,
        # print_data=True,
        test_with_connection=True,
        IM_windows=148,
        # OSC_stream=True,
        get_emg=True,
        get_gyro=True,
        get_accel=True,
        # OSC_ip="127.0.0.1",
    )
