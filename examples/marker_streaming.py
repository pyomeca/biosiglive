"""
This example shows how to retrieave markers data from a Vicon Nexus interface. Please note that the Vicon interface is
the only implemented method able to retrieve markers data.
First you have to create a ViconClient object. This object will be used to connect to the Vicon system and to retrieve the
data. Then you have to add a marker set to the interface. For now only one marker set can be added.
The marker set take the following arguments:
    - nb_markers: int
        Number of markers.
    - name: str
        Name of the markers set.
    - marker_names: Union[list, str]
        List of markers names.
    subject_name: str
        Name of the subject. If None, the subject will be the first one in Nexus.
    rate : int
        Rate of the camera set used to record marker trajectories.
    unit : str
        Unit of the marker trajectories ("mm" or "m").
If you want to display the markers in a 3D scatter plot you can add a Scatter3D plot to the interface. You can pass
 the size and tho color of the marker through the marker_kwargs argument. Please take a look at the documentation of
 the Scatter3D plot for more information.
Then the data streaming take place in a loop where the function get_marker_set_data() is used to retrieve the data from
the interface. The data is then passed to the plot through the update() method by a matrix of (n_frame, n_markers, 3).
 Where the plot parameters can be updated.
"""
from time import sleep, time
from custom_interface import MyInterface
from biosiglive import LivePlot, PlotType, ViconClient

if __name__ == "__main__":
    try_offline = True

    output_file_path = "trial_x.bio"
    if try_offline:
        interface = MyInterface(system_rate=100, data_path="abd.bio")
    else:
        interface = ViconClient(ip="localhost", system_rate=100)

    n_markers = 15
    interface.add_marker_set(
        nb_markers=n_markers, data_buffer_size=100, marker_data_file_key="markers", name="markers", rate=100, unit="mm"
    )
    marker_plot = LivePlot(name="markers", plot_type=PlotType.Scatter3D)
    marker_plot.init()
    time_to_sleep = 1 / 100
    offline_count = 0
    mark_to_plot = []
    while True:
        tic = time()
        mark_tmp, _ = interface.get_marker_set_data()
        marker_plot.update(mark_tmp[:, :, -1].T, size=0.03)
        loop_time = time() - tic
        real_time_to_sleep = time_to_sleep - loop_time
        if real_time_to_sleep > 0:
            sleep(real_time_to_sleep)
