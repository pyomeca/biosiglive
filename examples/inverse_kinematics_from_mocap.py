import time
from biosiglive import ViconClient, InverseKinematicsMethods, save, LivePlot, PlotType, InterfaceType
from custom_interface import MyInterface


if __name__ == "__main__":
    output_file_path = "trial_x.bio"
    interface = InterfaceType.Custom
    if interface == InterfaceType.Custom:
        interface = MyInterface(system_rate=100, data_path="abd.bio")
    else:
        interface = ViconClient(ip="localhost", system_rate=100)

    interface.add_marker_set(
        nb_markers=16,
        data_buffer_size=100,
        marker_data_file_key="markers",
        name="markers",
        rate=100,
        kinematics_method=InverseKinematicsMethods.BiorbdKalman,
        model_path="model/Wu_Shoulder_Model_mod_wt_wrapp.bioMod",
    )

    # Add plot
    marker_plot = LivePlot(name="markers", plot_type=PlotType.Skeleton)
    marker_plot.init(model_path="model/Wu_Shoulder_Model_mod_wt_wrapp.bioMod", show_floor=False, show_muscles=False)
    while True:
        tic = time.time()
        interface.get_marker_set_data()
        Q, Qdot = interface.marker_sets[0].get_kinematics(get_markers_data=True)
        marker_plot.update(Q)
        save({"Q": Q, "Qdot": Qdot}, output_file_path)
        if interface == InterfaceType.Custom:
            time.sleep((1 / 100) - (time.time() - tic))
