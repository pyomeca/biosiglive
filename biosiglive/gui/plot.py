import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
import pyqtgraph.widgets.RemoteGraphicsView as rgv
from PyQt5.QtWidgets import *
import numpy as np
from typing import Union
import matplotlib.pyplot as plt
from math import ceil


class Plot:
    def __init__(self, type: str = "curve"):
        self.type = None
        self.channel_names = None
        self.background_color = None
        self.figure_name = None
        self.type = type
        self.rplt = None
        self.box = None
        self.layout = None
        self.app = None

    def _plot_mvc(self, raw_data: np.ndarray, proc_data: np.ndarray, command: str, col: int = 4):
        """
                Plot data
                ----------
                raw_data: np.array()
                    raw data to plot of size (nb_muscles, nb_frames)
                proc_data : np.array()
                    processed data to plot of size (nb_muscles, nb_frames)
                command: str()
                    command to know which data to plot
                col: int
                    number of columns wanted in the plot.

                Returns
                -------
                """
        data = proc_data if command == "p" else raw_data
        plt.figure(self.try_name)
        size_police = 12
        col = self.n_muscles if self.n_muscles <= 4 else col
        line = ceil(self.n_muscles / col)
        for i in range(self.n_muscles):
            plt.subplot(line, col, i + 1)
            if i % 4 == 0:
                plt.ylabel("Activation level", fontsize=size_police)
            plt.plot(data[i, :], label="raw_data")
            if command == "b":
                plt.plot(proc_data[i, :], label="proc_data")
                plt.legend()
            plt.title(self.muscle_names[i], fontsize=size_police)
        plt.show()


class LivePlot:
    def __init__(self):
        self.multi_process = None
        self.check_box = True
        self.plot = []
        self.resize = (800, 800)
        self.move = (0, 0)
        self.progress_bar_max = 1000

    def add_new_plot(self, plot_name: str = "qt_app", plot_type="curve", channel_names: str = None):
        plot = Plot(type=plot_type)
        plot.channel_names = channel_names
        plot.figure_name = plot_name
        self.plot.append(plot)

    def init_plot_window(self, plot: Plot, use_checkbox: bool = True, remote: bool = True):
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

    def update_plot_window(self, plot: Plot, data: np.ndarray, app, rplt, box):
        if plot.type == "progress_bar":
            self._update_progress_bar(data, app, rplt, box)
        elif plot.type == "curve":
            self._update_curve(data, app, self.plot, plot.channel_names)
        else:
            raise ValueError(f"The plot type ({plot.type}) is not supported.")

    def _init_curve(self, figure_name: str = "Figure",
                    subplot_label: Union[list, str] = None,
                    nb_subplot: int = None,
                    checkbox: bool = True,
                    remote: bool = True):
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
        for i in range(data.shape[0]):
            value = np.mean(data[i, :])
            rplt[i].setValue(int(value))
            name = subplot_label[i] if subplot_label else f"plot_{i}"
            rplt[i].setFormat(f"{name}: {int(value)} {unit}")
        app.processEvents()

    @staticmethod
    def _init_layout(figure_name: str = "Figure", resize: tuple = (400, 400), move: tuple = (0, 0)):
        app = pg.mkQApp(figure_name)
        layout = pg.LayoutWidget()
        layout.resize(resize[0], resize[1])
        layout.move(move[0], move[1])
        return layout, app
