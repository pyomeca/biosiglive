"""
This file is part of biosiglive. It is used to plot the data in live or offline mode.
"""
try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtGui
    import pyqtgraph.widgets.RemoteGraphicsView as rgv
    from PyQt5.QtWidgets import *
except ModuleNotFoundError:
    pass
import numpy as np
from typing import Union
import matplotlib.pyplot as plt
from math import ceil
from ..enums import PlotType
import time


class LivePlot:
    def __init__(
        self,
        nb_subplots: int,
        plot_type: Union[PlotType, str] = PlotType.Curve,
        name: str = None,
        channel_names: list = None,
        rate: int = None,
        unit: str = None,
    ):
        """
        Initialize the plot class.

        Parameters
        ----------
        plot_type : Union[PlotType, str]
            type of the plot (curve, spectrogram, ...)
        name : str
            name of the plot
        channel_names : list
            list of the channel names
        nb_subplots : int
            number of subplots
        unit : str
            unit of the plot
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
        self.unit = unit
        self.rate = rate
        self.rplt = None
        self.box = None
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

    def init(
        self,
        plot_windows: Union[int, list] = None,
        use_checkbox: bool = True,
        remote: bool = True,
        bar_graph_max_value: Union[int, list] = None,
        msk_model: str = None,
        **kwargs,
    ):
        """
        This function is used to initialize the qt app.
        Parameters
        ----------
        plot_windows: Union[int, list]
            The number of frames ti plot. If is a list, the number of frames to plot for each subplot.
        use_checkbox: bool
            If True, the checkbox is used.
        remote: bool
            If True, the plot is done in a separate process.
        bar_graph_max_value: Union[int, list]
            The maximum value of the progress bar.
        **kwargs:
            The arguments of the bioviz plot.
        """
        self.msk_model = msk_model
        self.plot_buffer = [None] * self.nb_subplot
        if isinstance(plot_windows, int):
            plot_windows = [plot_windows] * self.nb_subplot
        self.plot_windows = plot_windows
        if self.plot_type == PlotType.Curve:
            self.rplt, self.layout, self.app, self.box = self._init_curve(
                self.figure_name, self.channel_names, self.nb_subplot, checkbox=use_checkbox
            )
        elif self.plot_type == PlotType.ProgressBar:
            self.rplt, self.layout, self.app = self._init_progress_bar(
                self.figure_name,
                self.nb_subplot,
                bar_graph_max_value,
            )

        elif self.plot_type == PlotType.Skeleton:
            if not self.msk_model:
                raise ValueError("Please provide the path to the model.")
            self.viz = self._init_skeleton(self.msk_model, **kwargs)

        else:
            raise ValueError(f"The plot type ({self.plot_type}) is not supported.")

    def update(self, data: Union[np.ndarray, list]):
        """
        This function is used to update the qt app.
        Parameters
        ----------
        data: Union[np.ndarray, list]
            The data to plot. if list, the data to plot for each subplot.
        """
        update = True
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
                    self.plot_buffer[i] = data[i]
                elif self.plot_buffer[i].shape[1] < self.plot_windows[i]:
                    self.plot_buffer[i] = np.append(self.plot_buffer[i], data[i], axis=1)
                elif self.plot_buffer[i].shape[1] >= self.plot_windows[i]:
                    size = data[i].shape[1]
                    self.plot_buffer[i] = np.append(self.plot_buffer[i][:, size:], data[i], axis=1)
                data[i] = self.plot_buffer[i]
        if self.rate and self.once_update:
            if 1 / (time.time() - self.last_plot) > self.rate:
                update = False
            else:
                update = True
        if update:
            self.once_update = True
            if self.plot_type == PlotType.ProgressBar:
                self._update_progress_bar(data, self.app, self.rplt, self.channel_names, self.unit)
            elif self.plot_type == PlotType.Curve:
                self._update_curve(data, self.app, self.rplt, self.box)
            elif self.plot_type == PlotType.Skeleton:
                self._update_skeleton(data, self.viz)
            else:
                raise ValueError(f"The plot type ({self.plot_type}) is not supported.")
            self.last_plot = time.time()

    def _init_curve(
        self,
        figure_name: str = "Figure",
        subplot_label: Union[list, str] = None,
        nb_subplot: int = None,
        checkbox: bool = True,
        # remote: bool = True,
    ):
        """
        This function is used to initialize the curve plot.
        Parameters
        ----------
        figure_name: str
            The name of the figure.
        subplot_label: str or list
            The label of the subplot.
        nb_subplot: int
            The number of subplot.
        checkbox: bool
            If True, the checkbox is used.
        remote: bool
            If True, the plot is done in a separate process.

        Returns
        -------
        app: QApplication
            The qt app.
        rplt: method
            The plot.
        layout:
            The layout of the qt app.
        box: QCheckBox
            The checkbox.
        """
        # TODO add remote statement
        # --- Curve graph --- #
        resize = self.resize
        move = self.move
        layout, app = self._init_layout(figure_name, resize, move)
        remote = []
        label = QtGui.QLabel()
        box = []
        rplt = []
        row_count = 0
        col_span = 4 if nb_subplot > 8 else 8

        if not isinstance(subplot_label, list):
            subplot_label = [subplot_label]
        if isinstance(subplot_label, list) and len(subplot_label) != nb_subplot:
            raise ValueError("The length of the subplot_label list must be equal to the number of subplot")

        for plot in range(nb_subplot):
            remote.append(rgv.RemoteGraphicsView())
            remote[plot].pg.setConfigOptions(antialias=True)
            app.aboutToQuit.connect(remote[plot].close)
            if checkbox:
                if subplot_label:
                    box.append(QtGui.QCheckBox(subplot_label[plot]))
                else:
                    box.append(QtGui.QCheckBox(f"plot_{plot}"))
            if plot >= 8:
                if checkbox:
                    layout.addWidget(box[plot], row=1, col=plot - 8)
                layout.addWidget(remote[plot], row=plot - 8 + 2, col=4, colspan=col_span)
            else:
                if checkbox:
                    layout.addWidget(box[plot], row=0, col=plot)
                layout.addWidget(remote[plot], row=plot + 2, col=0, colspan=col_span)
            rplt.append(remote[plot].pg.PlotItem())
            rplt[plot]._setProxyOptions(deferGetattr=True)  ## speeds up access to rplt.plot
            remote[plot].setCentralItem(rplt[plot])
            layout.addWidget(label)
            layout.show()
            row_count += 1
        return rplt, layout, app, box

    def _init_progress_bar(
        self, figure_name: str = "Figure", nb_subplot: int = None, bar_graph_max_value: Union[int, list] = 100
    ):
        """
        This function is used to initialize the curve plot.
        Parameters
        ----------
        figure_name: str
            The name of the figure.
        nb_subplot: int
            The number of subplot.

        Returns
        -------
        app: QApplication
            The qt app.
        rplt: Plot
            The plot.
        layout: pg.LayoutWidget
            The layout of the qt app.
        bar_graph_max_value: Union[int, list]
            The maximum value of the progress bar.
        """
        # --- Progress bar graph --- #
        layout, app = self._init_layout(figure_name, resize=self.resize, move=self.move)
        rplt = []
        row_count = 0
        if isinstance(bar_graph_max_value, int):
            bar_graph_max_value = [bar_graph_max_value] * nb_subplot
        for plot in range(nb_subplot):
            rplt.append(QProgressBar())
            rplt[plot].setMaximum(bar_graph_max_value[plot])
            layout.addWidget(rplt[plot], row=plot, col=0)
            layout.show()
            row_count += 1

        return rplt, layout, app

    @staticmethod
    def _update_curve(data: list, app, rplt: list, box: list):
        """
        This function is used to update the curve plot.
        Parameters
        ----------
        data: list
            The data to plot.
        app: QApplication
            The qt app.
        rplt: list
            The plot.
        box: QCheckBox
            The checkbox.

        Returns
        -------

        """
        for i in range(len(data)):
            if len(box) != 0:
                if box[i].isChecked() is True:
                    rplt[i].plot(data[i][0, :], clear=True, _callSync="off")
            else:
                rplt[i].plot(data[i][0, :], clear=True, _callSync="off")
        app.processEvents()

    @staticmethod
    def _update_progress_bar(data: list, app, rplt: list, subplot_label: list, unit: str = ""):
        """
        This function is used to update the progress bar plot.
        Parameters
        ----------
        data: list
            The data to plot.
        app: QApplication
            The qt app.
        rplt: list
            The plot.
        subplot_label: list
            The subplot label.
        unit: str
            The unit of the data to plot.
        """

        if subplot_label and len(subplot_label) != len(data):
            raise RuntimeError(
                f"The length of Subplot labels ({len(subplot_label)}) is different than"
                f" the first dimension of your data ({len(data)})."
            )

        for i in range(len(data)):
            value = np.mean(data[i][0, :])
            rplt[i].setValue(int(value))
            name = subplot_label[i] if subplot_label else f"plot_{i}"
            rplt[i].setFormat(f"{name}: {int(value)} {unit}")
        app.processEvents()

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
        data_mat = np.ndarray((len(data), data[0].shape[1]))
        for i, d in enumerate(data):
            data_mat[i, :] = d
        viz.set_q(data, refresh_window=True)

    def _init_skeleton(self, model_path: str, **kwargs):
        try:
            import bioviz
        except ImportError:
            raise ImportError("Please install bioviz (github.com/pyomeca/bioviz) to use the skeleton plot.")
        plot = bioviz.Viz(model_path, **kwargs)
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
            col = data[0].shape[0] if data[0].shape[0] <= 4 else nb_column
        line = ceil(data[0].shape[0] / col)
        for i in range(data[0].shape[0]):
            plt.subplot(line, col, i + 1)
            if y_label and i % 4 == 0:
                plt.ylabel(y_label, fontsize=size_police)
            if x_label:
                plt.xlabel(x_label, fontsize=size_police)
            for j in range(nb_data):
                x = x if x is not None else np.linspace(0, data[j].shape[1], data[j].shape[1])
                plt.plot(x, data[j][i, :], label=legend[j])
            plt.legend()
            if subplot_title:
                plt.title(subplot_title[i], fontsize=size_police)
        plt.show()
