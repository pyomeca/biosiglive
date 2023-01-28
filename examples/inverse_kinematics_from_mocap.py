"""
This example shows how to apply a inverse kinematics method to a biorbd model and a marker set.
To use this methods you will have to use and install the biorbd(https://github.com/pyomeca/biorbd)
 and bioviz libraries (https://github.com/pyomeca/bioviz). More informations are available on the github page of the
  projects. This example use a streaming of markers data from Vicon Nexus interface.
Note that a custom interface is also available from the example 'custom_interface.py' and it allow the user
to run the examples without any device connection by streaming data from a provided data file.
For more informations about the
way to do that please refer to marker_streaming.py example. Here the processing method used is the inverse kinematics
method from the biorbd library. The implemented methods are avalable in the InverseKinematicsMethods class. This method
take the biorbd model as argument in the initialization.
The process method will return the joint angle and joint velocity of the model. You can display the result using the
skeleton plot which use the bioviz library. In the initialization of the skeleton plot you can specify the arguments
belonging to the bioviz.Viz function. For more informations about the arguments please refer to the bioviz documentation.
Be aware that the skeleton plot might take some time and slow the loop. If you want to display the data in real time
think about use light .vtp file inside the model.
"""
import time
from biosiglive import ViconClient, InverseKinematicsMethods, LivePlot, PlotType, InterfaceType
from custom_interface import MyInterface


if __name__ == "__main__":
    interface = InterfaceType.Custom
    if interface == InterfaceType.Custom:
        interface = MyInterface(system_rate=100, data_path="abd.bio")
    else:
        interface = ViconClient(ip="127.0.0.1", system_rate=100)

    interface.add_marker_set(
        nb_markers=16,
        data_buffer_size=100,
        marker_data_file_key="markers",
        name="markers",
        rate=100,
        kinematics_method=InverseKinematicsMethods.BiorbdKalman,
        model_path="model/Wu_Shoulder_Model_mod_wt_wrapp.bioMod",
        unit="mm",
    )

    # Add plot
    marker_plot = LivePlot(name="markers", plot_type=PlotType.Skeleton)
    marker_plot.init(model_path="model/Wu_Shoulder_Model_mod_wt_wrapp.bioMod", show_floor=False, show_muscles=False)
    while True:
        tic = time.time()
        Q, Qdot = interface.get_kinematics_from_markers(marker_set_name="markers", get_markers_data=True)
        marker_plot.update(Q)
        if interface == InterfaceType.Custom:
            time.sleep((1 / 100) - (time.time() - tic))
