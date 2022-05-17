from biosiglive.interfaces.vicon_interface import ViconClient
from biosiglive.processing.data_processing import RealTimeProcessing
from biosiglive.gui.plot import LivePlot

if __name__ == '__main__':
    show_cadence = True
    vicon_interface = ViconClient()
    vicon_interface.add_device("treadmill", "generic_device", rate=2000)
    vicon_interface.devices[-1].set_processing_method(RealTimeProcessing().get_peaks)
    force_z, force_z_process = [], []
    if show_cadence:
        plot_app = LivePlot()
        plot_app.add_new_plot("cadence", "curve", "force_z")
        rplt, window, app, box = plot_app.init_plot_window(plot=plot_app.plot[0], use_checkbox=True)

    while True:
        data = vicon_interface.get_device_data("treadmill")
        force_z_tmp = data[0][1, :]
        cadence, force_z_process, force_z = vicon_interface.devices[0].process_method(new_sample=force_z_tmp,
                                                                                      signal=force_z,
                                                                                      signal_proc=force_z_process,
                                                                                      threshold=5,
                                                                                      window_len=200
                                                                                      )
        if show_cadence:
            plot_app.update_plot_window(plot_app.plot[0], force_z, app, rplt, box)
        print(f"Mean cadence for the last 10ms is :{cadence}")



