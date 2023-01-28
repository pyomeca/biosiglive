"""
This file contains all the plot functions to plot the data in live or offline.
"""
try:
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl
    from PyQt5.QtWidgets import QProgressBar
except ModuleNotFoundError:
    pass
import numpy as np
from typing import Union
import matplotlib.pyplot as plt
from math import ceil
from ..enums import PlotType
import time


# TODO: Add plot class to be able to plot several curves in the same plot. And do several plot in the same app.
class LivePlot:
    def __init__(
        self,
        nb_subplots: int = 1,
        plot_type: Union[PlotType, str] = PlotType.Curve,
        name: str = None,
        channel_names: list = None,
        rate: int = None,
    ):
        """
        Initialize the plot class.

        Parameters
        ----------
        plot_type : Union[PlotType, str]
            type of the plot (curve, progress bar, 3D scatter, skeleton...).
        name : str
            name of the plot
        channel_names : list
            list of the channel names (subplots names)
        nb_subplots : int
            number of subplots
        """
        if isinstance(plot_type, str):
            if plot_type not in [t.value for t in PlotType]:
                raise ValueError("Plot type not recognized")
            plot_type = PlotType(plot_type)
        self.plot_type = plot_type

        if nb_subplots and channel_names:
            if len(channel_names) != nb_subplots:
                raise ValueError("The number of subplots is not equal to the number of channel names")

        self.channel_names = channel_names
        self.figure_name = name
        self.nb_subplot = nb_subplots
        self.rate = rate
        self.layout = None
        self.app = None
        self.viz = None
        self.resize = (400, 400)
        self.move = (0, 0)
        self.plot_windows = None
        self.plot_buffer = None
        self.msk_model = None
        self.last_plot = None
        self.once_update = False
        self.plots = []
        self.curves = []
        self.ptr = []
        self.unit = ""
        self.size_to_append = []

    def init(
        self,
        plot_windows: Union[int, list] = None,
        **kwargs,
    ):
        """
        This function is used to initialize the qt app.

        Parameters
        ----------
        plot_windows: Union[int, list]
            The number of frames ti plot. If is a list, the number of frames to plot for each subplot.
        """
        self.plot_buffer = [None] * self.nb_subplot
        if isinstance(plot_windows, int):
            plot_windows = [plot_windows] * self.nb_subplot
        self.plot_windows = plot_windows
        if self.plot_type == PlotType.Curve:
            self._init_curve(self.figure_name, self.channel_names, self.nb_subplot, **kwargs)
        elif self.plot_type == PlotType.ProgressBar:
            self._init_progress_bar(self.figure_name, self.nb_subplot, **kwargs)
        elif self.plot_type == PlotType.Scatter3D:
            if self.nb_subplot != 1:
                raise ValueError("The number of subplots should be 1 for 3DScatter plot.")
            self._init_3d_scatter(self.figure_name, **kwargs)

        elif self.plot_type == PlotType.Skeleton:
            self.viz = self._init_skeleton(**kwargs)

        else:
            raise ValueError(f"The plot type ({self.plot_type}) is not supported.")

    def update(self, data: Union[np.ndarray, list], **kwargs):
        """
        This function is used to update the qt app.

        Parameters
        ----------
        data: Union[np.ndarray, list]
            The data to plot. If it is a list, the data to plot for each subplot.
        """
        update = True
        if self.plot_type != PlotType.Scatter3D and self.plot_type != PlotType.Skeleton:
            if isinstance(data, list):
                if len(data) != self.nb_subplot:
                    raise ValueError("The number of subplots is not equal to the number of data.")
                for d in data:
                    if isinstance(d, np.ndarray):
                        if len(d.shape) != 2:
                            raise ValueError("The data should be a 2D array.")
                    else:
                        raise ValueError("The data should be a 2D array.")
            if isinstance(data, np.ndarray):
                data_mat = data
                data = []
                if data_mat.shape[0] != self.nb_subplot:
                    raise ValueError("The number of subplots is not equal to the number of data.")
                for d in data_mat:
                    data.append(d[np.newaxis, :])

        if self.plot_windows:
            for i in range(self.nb_subplot):
                if self.plot_buffer[i] is None:
                    self.plot_buffer[i] = data[i][..., -self.plot_windows[i] :]
                    if self.plot_buffer[i].shape[1] < self.plot_windows[i]:
                        size = self.plot_windows[i] - self.plot_buffer[i].shape[1]
                        self.plot_buffer[i] = np.append(
                            np.zeros((self.plot_buffer[i].shape[0], size)), self.plot_buffer[i], axis=-1
                        )
                elif self.plot_buffer[i].shape[1] < self.plot_windows[i]:
                    self.plot_buffer[i] = np.append(self.plot_buffer[i], data[i], axis=-1)
                elif self.plot_buffer[i].shape[1] >= self.plot_windows[i]:
                    size = data[i].shape[1]
                    self.plot_buffer[i] = np.append(self.plot_buffer[i][..., size:], data[i], axis=-1)
            data = self.plot_buffer
        if self.rate and self.once_update:
            plot_time = time.time() - self.last_plot
            if plot_time != 0 and 1 / plot_time > self.rate:
                update = False
            else:
                update = True
        if update:
            self.once_update = True
            if self.plot_type == PlotType.ProgressBar:
                self._update_progress_bar(data)
            elif self.plot_type == PlotType.Curve:
                self._update_curve(data)
            elif self.plot_type == PlotType.Skeleton:
                self._update_skeleton(data, self.viz)
            elif self.plot_type == PlotType.Scatter3D:
                self._update_3d_scatter(data, **kwargs)
            else:
                raise ValueError(f"The plot type ({self.plot_type}) is not supported.")
            self.last_plot = time.time()

    def _init_curve(
        self,
        figure_name: str = "Figure",
        subplot_labels: Union[list, str] = None,
        nb_subplot: int = None,
        x_labels: Union[list, str] = None,
        y_labels: Union[list, str] = None,
        grid: bool = True,
        colors: Union[list, tuple] = None,
    ):
        """
        This function is used to initialize the curve plot.

        Parameters
        ----------
        figure_name: str
            The name of the figure.
        subplot_labels: Union[list, str]
            The labels of the subplots.
        nb_subplot: int
            The number of subplot.
        x_labels: Union[list, str]
            The labels of the x axis.
        y_labels: Union[list, str]
            The labels of the y axis.
        grid: bool
            If True, the grid is displayed.
        colors: Union[list, tuple]
            The colors of the curves.
        """
        # --- Curve graph --- #
        self.app = pg.mkQApp("Curve_plot")
        pg.setConfigOption("background", "w")
        pg.setConfigOption("foreground", "k")
        self.win = pg.GraphicsLayoutWidget(show=True)
        self.win.setWindowTitle(figure_name)
        nb_line = 4
        nb_col = ceil(nb_subplot / nb_line)
        line_count = 0
        self.win.resize(self.resize[0], self.resize[1])
        self.win.move(self.move[0], self.move[1])
        if colors:
            if isinstance(colors, tuple):
                colors = [colors] * nb_subplot
            elif isinstance(colors, list):
                if len(colors) != nb_subplot:
                    raise ValueError("The number of colors is not equal to the number of subplots.")
        else:
            colors = [(0, 128, 232)] * nb_subplot  # Blue
        if not x_labels:
            x_labels = ["Frames"] * nb_subplot
        else:
            if isinstance(x_labels, str):
                x_labels = [x_labels] * nb_subplot
            elif isinstance(x_labels, list):
                if len(x_labels) != nb_subplot:
                    raise ValueError("The number of x labels is not equal to the number of subplots.")

        if not y_labels:
            y_labels = ["Amplitude"] * nb_subplot
        else:
            if isinstance(y_labels, str):
                y_labels = [y_labels] * nb_subplot
            elif isinstance(y_labels, list):
                if len(y_labels) != nb_subplot:
                    raise ValueError("The number of y labels is not equal to the number of subplots.")
        if not subplot_labels:
            subplot_labels = [f"Subplot {i}" for i in range(nb_subplot)]
        else:
            if isinstance(subplot_labels, list):
                if len(subplot_labels) != nb_subplot:
                    raise ValueError("The number of subplot labels is not equal to the number of subplots.")
        for subplot in range(nb_subplot):
            self.ptr.append(0)
            self.size_to_append.append(0)
            if line_count == nb_col:
                self.win.nextRow()
                line_count = 0
            self.plots.append(self.win.addPlot(title=subplot_labels[subplot]))
            self.plots[-1].setDownsampling(mode="peak")
            self.plots[-1].setClipToView(False)
            self.curves.append(self.plots[-1].plot([], pen=colors[subplot], name="Blue curve"))
            self.plots[-1].setLabel("bottom", x_labels[subplot])
            self.plots[-1].setLabel("left", y_labels[subplot])
            self.plots[-1].showGrid(x=grid, y=grid)
            line_count += 1

    def _init_progress_bar(
        self,
        figure_name: str = "Figure",
        nb_subplot: int = None,
        bar_graph_max_value: Union[int, list] = 100,
        unit: Union[str, list] = "",
    ):
        """
        This function is used to initialize the curve plot.

        Parameters
        ----------
        figure_name: str
            The name of the figure.
        nb_subplot: int
            The number of subplot.
        bar_graph_max_value: int or list
            The maximum value of the bar graph.
        unit: str or list
            The unit of the bar graph.
        """
        # --- Progress bar graph --- #
        if isinstance(unit, str):
            self.unit = [unit] * nb_subplot
        self.layout, self.app = self._init_layout(figure_name, resize=self.resize, move=self.move)
        row_count = 0
        if bar_graph_max_value is None:
            bar_graph_max_value = [100] * nb_subplot
        if isinstance(bar_graph_max_value, int):
            bar_graph_max_value = [bar_graph_max_value] * nb_subplot
        for plot in range(nb_subplot):
            self.plots.append(QProgressBar())
            self.plots[-1].setMaximum(bar_graph_max_value[plot])
            self.layout.addWidget(self.plots[-1], row=plot, col=0)
            self.layout.show()
            row_count += 1

    def _init_3d_scatter(
        self,
        figure_name: str = "Figure",
        colors: Union[list, tuple] = (1.0, 0.0, 0.0, 0.5),
        size: Union[int, list] = 0.03,
    ):
        """
        This function is used to initialize the 3d scatter plot.

        Parameters
        ----------
        figure_name: str
            The name of the figure.
        colors: Union[list, tuple]
            The color of the scatter.
        size: Union[int, list]
            The size of the scatters.
        """
        # --- 3D scatter graph --- #
        self.app = pg.mkQApp("3D_scatter_plot")
        w = gl.GLViewWidget()
        w.opts["bgcolor"] = (0.2, 0.2, 0.2, 10)
        w.opts["distance"] = 8
        w.show()
        w.setWindowTitle(figure_name)
        g = gl.GLGridItem()
        # g.setColor((1, 1, 1, 100))
        w.addItem(g)
        pos = np.zeros((1, 3))
        self.plots.append(gl.GLScatterPlotItem(pos=pos, color=colors, size=size, pxMode=False))
        w.addItem(self.plots[-1])

    def _update_3d_scatter(
        self,
        data: Union[np.ndarray, list],
        colors: Union[list, tuple] = (0, 1.0, 0.0, 50),
        size: Union[list, float] = 0.03,
    ):
        """
        This function is used to update the 3d scatter plot.

        Parameters
        ----------
        data: np.ndarray
            The data to plot. (N, 3)
        colors: Union[list, tuple]
            The color of the scatter.
        size: float
            The size of the scatter.
        """
        if isinstance(data, np.ndarray):
            if len(data.shape) != 2:
                raise ValueError("The data must be a 2D array.")
            if data.shape[1] != 3:
                raise ValueError("The data must be a (N, 3) array.")
        if isinstance(colors, list):
            if len(colors) != len(data):
                raise ValueError("The number of colors is not equal to the number of data.")
        if isinstance(size, list):
            if len(size) != len(data):
                raise ValueError("The number of size is not equal to the number of data.")
        for plot in self.plots:
            plot.setData(pos=data, color=colors, size=size)
        self.app.processEvents()

    def _update_curve(self, data: list):
        """
        This function is used to update the curve plot.

        Parameters
        ----------
        data: list
            The data to plot.
        """
        if len(data) != len(self.curves):
            raise ValueError(
                f"The number of data ({len(data)}) is different from the number of curves ({len(self.curves)})."
            )
        for i in range(len(data)):
            if self.ptr[i] == 0:
                self.size_to_append[i] = data[i].shape[1]
            self.ptr[i] += self.size_to_append[i] * 2
            self.curves[i].setData(data[i][0, :])
            # self.curves[i].setPos(self.ptr[i], 0)
        self.app.processEvents()

    def _update_progress_bar(self, data: list):
        """
        This function is used to update the progress bar plot.

        Parameters
        ----------
        data: list
            The data to plot.
        """

        if self.channel_names and len(self.channel_names) != len(data):
            raise RuntimeError(
                f"The length of Subplot labels ({len(self.channel_names)}) is different than"
                f" the first dimension of your data ({len(data)})."
            )

        for i in range(len(data)):
            value = np.mean(data[i][0, :])
            self.plots[i].setValue(int(value))
            name = self.channel_names[i] if self.channel_names else f"plot_{i}"
            self.plots[i].setFormat(f"{name}: {int(value)} {self.unit[i]}")
        self.app.processEvents()

    @staticmethod
    def _update_skeleton(data: list, viz):
        """
        This function is used to update the skeleton plot.

        Parameters
        ----------
        data  : list
            The data to plot. list of length degree of freedom.
        viz: Viz3D
            The plot.

        Returns
        -------

        """
        viz.set_q(data[:, -1], refresh_window=True)

    @staticmethod
    def _init_skeleton(**kwargs):
        try:
            import bioviz
        except ImportError:
            raise ImportError("Please install bioviz (github.com/pyomeca/bioviz) to use the skeleton plot.")
        if not "model_path" in kwargs or "model" in kwargs:
            raise ValueError(
                "You must provide a model_path or a model to use the skeleton plot through"
                " the keyword arguments 'model_path' or 'model' respectively."
            )
        plot = bioviz.Viz(**kwargs)
        return plot

    @staticmethod
    def _init_layout(figure_name: str = "Figure", resize: tuple = (400, 400), move: tuple = (0, 0)):
        """
        This function is used to initialize the qt app layout.

        Parameters
        ----------
        figure_name: str
            The name of the figure.
        resize: tuple
            The size of the figure.
        move: tuple
            The position of the figure.

        Returns
        -------
        layout: QVBoxLayout
            The layout of the qt app.
        app: QApplication
            The qt app.

        """
        app = pg.mkQApp(figure_name)
        layout = pg.LayoutWidget()
        layout.resize(resize[0], resize[1])
        layout.move(move[0], move[1])
        return layout, app

    def disconnect(self):
        self.app.disconnect()
        try:
            self.app.closeAllWindows()
        except RuntimeError:
            pass


class OfflinePlot:
    """
    This class is used to plot data offline.
    """

    @staticmethod
    def multi_plot(
        data: Union[list, np.ndarray],
        x: Union[list, np.ndarray] = None,
        nb_column: int = None,
        y_label: str = None,
        x_label: str = None,
        legend: Union[list, str] = None,
        subplot_title: Union[str, list] = None,
        figure_name: str = None,
    ):
        """
        This function is used to plot multiple data in one figure.

        Parameters
        ----------
        data: list or np.ndarray
            The data to plot.
        x: list or np.ndarray
            The x-axis data.
        nb_column: int
            The number of columns in the figure.
        y_label: str
            The y-axis label.
        x_label: str
            The x-axis label.
        legend: list or str
            The legend of the data.
        subplot_title: str or list
            The title of the subplot.
        figure_name: str
            The name of the figure.
        """

        if not isinstance(data, list):
            data = [data]
        nb_data = len(data)
        plt.figure(figure_name)
        size_police = 12
        if nb_column:
            col = nb_column
        else:
            col = data[0].shape[0] if data[0].shape[0] <= 4 else 4
        line = ceil(data[0].shape[0] / col)
        if isinstance(legend, str):
            legend = [legend]
        for i in range(data[0].shape[0]):
            plt.subplot(line, col, i + 1)
            if y_label and i % 4 == 0:
                plt.ylabel(y_label, fontsize=size_police)
            if x_label:
                plt.xlabel(x_label, fontsize=size_police)
            for j in range(nb_data):
                if legend:
                    legend_tmp = legend[j]
                else:
                    legend_tmp = None
                if x is not None:
                    plt.plot(x, data[j][i, :], label=legend_tmp)
                else:
                    plt.plot(data[j][i, :], label=legend_tmp)

            plt.legend()
            if subplot_title:
                plt.title(subplot_title[i], fontsize=size_police)
        plt.show()
