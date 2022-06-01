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


class Plot:
    """
    This class is used to plot the data offline.
    """
    def multi_plot(self, data: Union[list, np.ndarray], x: Union[list, np.ndarray] = None,
                   nb_column: int = None, y_label: str = None, x_label: str = None,
                   legend: Union[list, str] = None, subplot_title: Union[str, list] = None, figure_name: str = None):
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
        # plt.tight_layout()
        plt.show()


class LivePlot:
    """
    This class is used to plot the data in live mode.
    """
    def __init__(self, multi_process: bool = False):
        """
        This function is used to initialize the LivePlot class.
        Parameters
        ----------
        multi_process: bool
            If True, the plot is done in a separate process.
        """
        self.multi_process = multi_process
        self.check_box = True
        self.plot = []
        self.resize = (800, 800)
        self.move = (0, 0)
        self.progress_bar_max = 1000
        self.type = "curve"

    def add_new_plot(self, plot_name: str = "qt_app", plot_type="curve", channel_names: Union[str, list] = None):
        """
        This function is used to add a new plot.
        Parameters
        ----------
        plot_name: str
            The name of the plot.
        plot_type: str
            The type of the plot.
        channel_names: str or list
            The name of the channels.
        """

        plot = LivePlot()
        plot.type = plot_type
        plot.channel_names = channel_names
        plot.figure_name = plot_name
        self.plot.append(plot)

    def init_plot_window(self, plot, use_checkbox: bool = True, remote: bool = True):
        """
        This function is used to initialize the qt app.
        Parameters
        ----------
        plot: method
            The plot to initialize.
        use_checkbox: bool
            If True, the checkbox is used.
        remote: bool
            If True, the plot is done in a separate process.

        Returns
        -------
        app: QApplication
            The qt app.
        rplt: Plot
            The plot.
        layout: QVBoxLayout
            The layout of the qt app.
        box: QCheckBox
            The checkbox.
        """
        if plot.type == "curve":
            rplt, layout, app, box = self._init_curve(plot.figure_name,
                                                      plot.channel_names,
                                                      len(plot.channel_names),
                                                      checkbox=use_checkbox,
                                                      remote=remote)
            return rplt, layout, app, box
        elif plot.type == "progress_bar":
            rplt, layout, app = LivePlot._init_progress_bar(plot.figure_name, plot.channel_names)
            return rplt, layout, app
        else:
            raise ValueError(f"The plot type ({plot.type}) is not supported.")

    def update_plot_window(self, plot, data: np.ndarray, app, rplt, box):
        """
        This function is used to update the qt app.
        Parameters
        ----------
        plot: method
            The plot to update.
        data: np.ndarray
            The data to plot.
        app: QApplication
            The qt app.
        rplt: method
            qt rplt.
        box: QCheckBox
            The checkbox.
        """

        if plot.type == "progress_bar":
            self._update_progress_bar(data, app, rplt, box)
        elif plot.type == "curve":
            self._update_curve(data, app, rplt, box)
        else:
            raise ValueError(f"The plot type ({plot.type}) is not supported.")

    def _init_curve(self, figure_name: str = "Figure",
                    subplot_label: Union[list, str] = None,
                    nb_subplot: int = None,
                    checkbox: bool = True,
                    remote: bool = True):
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
        layout, app = LivePlot._init_layout(figure_name, resize, move)
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

    def _init_progress_bar(self, figure_name: str = "Figure", nb_subplot: int = None):
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
        """
        # --- Progress bar graph --- #
        layout, app = LivePlot._init_layout(figure_name, resize=self.resize, move=self.move)
        rplt = []
        row_count = 0
        for plot in range(nb_subplot):
            rplt.append(QProgressBar())
            rplt[plot].setMaximum(self.progress_bar_max)
            layout.addWidget(rplt[plot], row=plot, col=0)
            layout.show()
            row_count += 1

        return rplt, layout, app

    @staticmethod
    def _update_curve(data: np.ndarray, app, rplt: list, box: list):
        """
        This function is used to update the curve plot.
        Parameters
        ----------
        data: np.ndarray
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
        for i in range(data.shape[0]):
            if len(box) != 0:
                if box[i].isChecked() is True:
                    if len(data.shape) == 2:
                        rplt[i].plot(data[i, :], clear=True, _callSync='off')
                    else:
                        raise ValueError("The data must be a 2D array.")
            else:
                if len(data.shape) == 2:
                    rplt[i].plot(data[i, :], clear=True, _callSync='off')
                else:
                    raise ValueError("The data must be a 2D array.")
        app.processEvents()

    @staticmethod
    def _update_progress_bar(data: np.ndarray, app, rplt: list, subplot_label: list, unit: str = ""):
        """
        This function is used to update the progress bar plot.
        Parameters
        ----------
        data: np.ndarray
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
        for i in range(data.shape[0]):
            value = np.mean(data[i, :])
            rplt[i].setValue(int(value))
            name = subplot_label[i] if subplot_label else f"plot_{i}"
            rplt[i].setFormat(f"{name}: {int(value)} {unit}")
        app.processEvents()

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
