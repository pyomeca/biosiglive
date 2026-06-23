""" Example that takes control of QTM, streams data etc """

import asyncio
import logging
import xml.etree.ElementTree as ET
import qtm_rt
import time
import numpy as np
from biosiglive import LivePlot, PlotType, QualisysClient

LOG = logging.getLogger("example")


async def setup():
    """ main function """

    connection = await qtm_rt.connect("192.168.254.1")

    if connection is None:
        return -1


    # Plot initialisation
    marker_plot = LivePlot(name="markers", plot_type=PlotType.Scatter3D)
    marker_plot.init()


    all_mks_data = np.zeros((3, 4, 1))
    time_to_sleep = 1 / 100

    while True:
        tic = asyncio.get_event_loop().time()
        packet = await connection.get_current_frame(components=['3d'])
        headermks, mks = packet.get_3d_markers()
        if mks is None:
            break

        LOG.info("Framenumber %s", packet.framenumber)


        # get_marker_data

        LOG.info("Component info: %s", headermks)
        mks_currentframe = np.zeros((3, headermks.marker_count, 1))

        for i, mks in enumerate(mks, 1):
            LOG.info("Marqueur %d", i)
            j = 0
            for marker in mks:
                LOG.info("\t%s", marker)
                mks_currentframe[j, i - 1, :] = marker / 1000
                j = j + 1

            all_mks_data = np.append(all_mks_data, mks_currentframe, axis=2)


        # plot actualisation
        marker_plot.update(all_mks_data[:, :, -1].T, size=0.1)
        loop_time = asyncio.get_event_loop().time() - tic
        real_time_to_sleep = time_to_sleep - loop_time
        if real_time_to_sleep > 0:
            await asyncio.sleep(real_time_to_sleep)



if __name__ == "__main__":
    asyncio.run(setup())
