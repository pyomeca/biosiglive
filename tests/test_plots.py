from biosiglive import OfflinePlot
import numpy as np
import matplotlib

matplotlib.use("Agg")


def test_offline_plot():
    np.random.seed(50)
    data = np.random.random((5, 20))
    OfflinePlot.multi_plot(data)


# TODO add test live plot
