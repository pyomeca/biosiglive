# Live plots

This example shows how to use the plot in real time inside a loop.
Here three different plots are used:
-Curve plot : allows to plot a curve on a number of frame plot_windows.
-progress barplot : Plot a progress bar with a maximum value to fit the filled area in the bar. A unit can also be sent to the function to improve the plot.
-3D scatter plot: Plots the scatter in a 3D space. The size and color can be changed using the size and color arguments, respectively.
The initialization is the same for each plot. The number of subplots can be specified as well as their names. The type of plot must be specified using the PlotType enum class. Finally, the targeted plot frequency can be defined, it will slow down the update and can therefore improve performance or visibility (for example, plotting bars at 100 Hz is difficult to understand).
Once initialized each plot can be updated in a loop using some data (here random ones).

```
import numpy as np
from time import time, sleep
from biosiglive import LivePlot, PlotType


if __name__ == "__main__":
    plot_curve = LivePlot(
        name="curve",
        rate=100,
        plot_type=PlotType.Curve,
        nb_subplots=4,
        channel_names=["1", "2", "3", "4"],
    )
    plot_curve.init(plot_windows=1000, y_labels=["Strikes", "Strikes", "Force (N)", "Force (N)"])
    plot = LivePlot(
        name="bar",
        rate=10,
        plot_type=PlotType.ProgressBar,
        nb_subplots=4,
        channel_names=["1", "2", "3", "4"],

    )
    plot.init(bar_graph_max_value=100, unit="N")
    plot_scatter = LivePlot(
        name="scatter",
        rate=50,
        plot_type=PlotType.Scatter3D,
    )
    plot_scatter.init()
    while True:
        tic = time()
        data = np.random.rand(3, 4, 1)*100
        plot.update(data[0, :, :])
        plot_curve.update(data[0, :, :])
        plot_scatter.update(data[:, :, -1].T/100, size=0.1)
        sleep((1 / 100))
```
