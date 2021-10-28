from biosiglive.server import Server
import os

if __name__ == "__main__":
    parent = os.path.dirname(os.getcwd())
    offline_path = parent + 'test_wt_connection.mat'
    with_connection = True
    # IP_server = '192.168.1.211'
    IP_server = "localhost"
    device_ip = "192.168.1.211"
    server_port = [50000, 50001]

    # Set program variables
    read_freq = 100
    n_electrode = 10

    # Set file to save data
    output_file = "stream_data_xxx"
    output_dir = "test_stream"

    # Run streaming data
    muscles_idx = (0, n_electrode - 1)
    MVC_list = [0, 0, 2]
    server = Server(
        IP=IP_server,
        server_ports=server_port,
        device="vicon",
        type="TCP",
        muscle_range=muscles_idx,
        device_host_ip=device_ip,
        acquisition_rate=read_freq,
        output_dir=output_dir,
        output_file=output_file,
        offline_file_path=offline_path
    )

    server.run(
        stream_emg=True,
        stream_markers=False,
        stream_imu=True,
        optim=True,
        plot_emg=False,
        norm_emg=True,
        test_with_connection=with_connection,
        mvc_list=MVC_list,
    )
