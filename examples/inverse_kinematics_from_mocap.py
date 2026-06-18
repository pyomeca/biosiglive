"""
This example shows how to apply an inverse kinematics method to a biorbd model and a set of markers.
To use these methods, you will need to use and install the biorbd(https://github.com/pyomeca/biorbd) and bioviz (https://github.com/pyomeca/bioviz) libraries. More information is available on the projects GitHub page. This example uses a marker data stream from the Vicon Nexus interface.
If you want to try this example offline, you can use the provided custom interface, called 'MyInterface', which you can use as a standard interface.
For more information on how to do this, please refer to the marker_streaming.py example. Here, the processing method used is the inverse kinematics method of the biorbd library. The implemented methods are available in the class InverseKinematicsMethods. This method takes the biorbd model as an argument in the initialization.
The processing method will return the angle and velocity of the model joint. You can display the result using the skeleton plot which uses the bioviz library. In the skeleton plot initialization, you can specify the arguments belonging to the bioviz.Viz function. For more information about the arguments, please refer to the bioviz documentation.
Be aware that the skeleton plot may take some time and slow down the loop. If you want to display the data in real time, consider using a lightweight *.vtp file inside the model.
"""

import time

import numpy as np

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

    marker_plot = LivePlot(name="markers", plot_type=PlotType.Skeleton)
    marker_plot.init(model_path="model/Wu_Shoulder_Model_mod_wt_wrapp.bioMod", show_floor=False, show_muscles=False)
    while True:
        tic = time.time()
        Q, Qdot = interface.get_kinematics_from_markers(marker_set_name="markers", get_markers_data=True)
        marker_plot.update(Q)
        if interface == InterfaceType.Custom:
            time.sleep((1 / 100) - (time.time() - tic))
