""" Example that takes control of QTM, streams data etc """

import asyncio
import logging
import xml.etree.ElementTree as ET
import qtm_rt
import time
import numpy as np
from biosiglive import LivePlot, PlotType, QualisysClient
LOG = logging.getLogger("example")

"""
async def package_receiver(queue):
     Asynchronous function that processes queue until None is posted in queue
    LOG.info("Entering package_receiver")
    while True:
        packet = await queue.get()
        if packet is None:
            break

        LOG.info("Framenumber %s", packet.framenumber)


        latency = (packet.timestamp) - time.time()
        LOG.info("Latency: %s", latency)
  

        headermks, mks = packet.get_3d_markers()
        LOG.info("Component info: %s", headermks)
        for i, mks in enumerate(mks, 1):
            LOG.info("Marqueur %d", i)
            for marker in mks:
                LOG.info("\t%s", marker)

        headerf, forces = packet.get_force()
        LOG.info("Component info: %s", headerf)
        for i, forces in enumerate(forces, 1):
            LOG.info("Force %d", i)
            for force in forces:
                LOG.info("\t%s", force)

        marker_plot.update(mks[:, :, -1].T, size=0.1)

    LOG.info("Exiting package_receiver")
"""


async def setup():
    """ main function """

    connection = await qtm_rt.connect("192.168.254.1")

    if connection is None:
        return -1

    #General information (~add_markerset & add_forceplate)
    xlm_general = await connection.get_parameters(parameters=['general'])
    GeneralInfo = ET.fromstring(xlm_general)
    Fs = int(GeneralInfo.find('.//Frequency').text)

    xlm_3d = await connection.get_parameters(parameters=['3d'])
    MksInfo = ET.fromstring(xlm_3d)
    nb_mks = MksInfo.find('.//Labels').text
    nom_mks = [label.find('Name').text for label in MksInfo.findall(".//Label")]
    print(Fs)
    print(nom_mks)

    xlm_force = await connection.get_parameters(parameters=['force'])
    ForceInfo = ET.fromstring(xlm_force)
    unit_forceplate = ForceInfo.find('.//Unit_Force').text
    nom_forceplate = [label.find('Name').text for label in ForceInfo.findall(".//Plate")]
    print(nom_forceplate)
    print(unit_forceplate)

    # Plot initialisation
    marker_plot = LivePlot(name="markers", plot_type=PlotType.Scatter3D)
    marker_plot.init()
    """
    force1_plot = LivePlot(
        name="force", rate=1000, plot_type=PlotType.Curve, nb_subplots=3
    )
    force1_plot.init(plot_windows=1000, y_labels="Force (N)")

    force2_plot = LivePlot(
        name="force", rate=1000, plot_type=PlotType.Curve, nb_subplots=3
    )
    force2_plot.init(plot_windows=1000, y_labels="Force (N)")
    """
    #Data  extraction
    queue = asyncio.Queue()

    #receiver_future = asyncio.ensure_future(package_receiver(queue))

    await connection.stream_frames(components=["6d", "3d", "force"], on_packet=queue.put_nowait)

    all_forcedata = np.zeros((3, 2, 1))
    all_mks_data = np.zeros((3, 4, 1))
    time_to_sleep = 1 / 200
    while True:
        tic=asyncio.get_event_loop().time()
        packet = await queue.get()
        if packet is None:
            break

        LOG.info("Framenumber %s", packet.framenumber)

        """
        latency = (packet.timestamp) - time.time()
        LOG.info("Latency: %s", latency)
        """
        #get_marker_data
        headermks, mks = packet.get_3d_markers()
        LOG.info("Component info: %s", headermks)
        mks_currentframe = np.zeros((3, headermks.marker_count, 1))

        for i, mks in enumerate(mks, 1):
            LOG.info("Marqueur %d", i)
            j=0
            for marker in mks:
                LOG.info("\t%s", marker)
                mks_currentframe[j, i - 1, :] = marker/1000
                j=j+1

            all_mks_data = np.append(all_mks_data, mks_currentframe, axis=2)

        """
        # get_force_data
        headerf, forces = packet.get_force()
        LOG.info("Component info: %s", headerf)
        force_currentframe = np.zeros((9, headerf.plate_count, 1))
        for i, forces in enumerate(forces, 1):
            LOG.info("Force %d", i)
            for force in forces:
                LOG.info("\t%s", force)

            forcedata = forces[i-1][1][1]
            forces_data_tmp = [forcedata.x, forcedata.y, forcedata.z,
                               forcedata.x_m, forcedata.y_m, forcedata.z_m,
                               forcedata.x_a, forcedata.y_a, forcedata.z_a]
            force_currentframe[:, i-1,:] = np.array(forces_data_tmp)[:, np.newaxis]

        all_forcedata = np.append(all_forcedata, force_currentframe, axis=2)
        """
        #plot actualisation
        marker_plot.update(all_mks_data[:, :, -1].T, size=0.1)
        loop_time = asyncio.get_event_loop().time() - tic
        real_time_to_sleep = time_to_sleep - loop_time
        if real_time_to_sleep > 0:
            await asyncio.sleep(real_time_to_sleep)
        """
        force1_plot.update(np.array(all_forcedata[0:3, 0, -1:]))
        force2_plot.update(np.array(all_forcedata[0:3, 1, -1:]))
        """
#        asyncio.ensure_future(shutdown(30, connection, receiver_future, queue))


if __name__ == "__main__":
    asyncio.run(setup())
    """
    loop = asyncio.get_event_loop()
    asyncio.ensure_future(setup())
    loop.run_forever()
    """