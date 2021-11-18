import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
import pyqtgraph.widgets.RemoteGraphicsView as rgv
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

try:
    from optim_funct import markers_fun
    biorbd_module = True
except ModuleNotFoundError:
    biorbd_module = False
import numpy as np

if biorbd_module is True:

    def plot_results(
        biorbd_model,
        X_est,
        q_ref,
        Ns,
        rt_ratio,
        nbQ,
        dq_ref,
        U_est,
        u_ref,
        nbGT,
        muscles_target,
        force_est,
        force_ref,
        markers_target,
        use_torque,
    ):
        plt.figure("q")
        for i in range(biorbd_model.nbQ()):
            plt.subplot(3, 2, i + 1)
            plt.plot(X_est[i, :], "x")
            plt.plot(q_ref[i, 0 : Ns + 1 : rt_ratio])
        plt.legend(labels=["q_est", "q_ref"], bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)

        plt.figure("qdot")
        for i in range(biorbd_model.nbQ(), biorbd_model.nbQ() * 2):
            plt.subplot(3, 2, i - nbQ + 1)
            plt.plot(X_est[i, :], "x")
            plt.plot(dq_ref[i - nbQ, 0 : Ns + 1 : rt_ratio])
        plt.legend(labels=["q_est", "q_ref"], bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
        if use_torque:
            plt.figure("Tau")
            for i in range(biorbd_model.nbQ()):
                plt.subplot(3, 2, i + 1)
                plt.plot(U_est[i, :], "x")
                plt.plot(u_ref[i, 0 : Ns + 1 : rt_ratio])
                plt.plot(muscles_target[i, :], "k--")
            plt.legend(labels=["Tau_est", "Tau_ref"], bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)

        plt.figure("Muscles excitations")
        for i in range(biorbd_model.nbMuscles()):
            plt.subplot(4, 5, i + 1)
            plt.plot(U_est[nbGT + i, :])
            plt.plot(u_ref[nbGT + i, 0:Ns:rt_ratio], c="red")
            plt.plot(muscles_target[nbGT + i, 0:Ns:rt_ratio], "k--")
            plt.title(biorbd_model.muscleNames()[i].to_string())
        plt.legend(
            labels=["u_est", "u_ref", "u_with_noise"], bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0
        )

        plt.figure("Muscles_force")
        for i in range(biorbd_model.nbMuscles()):
            plt.subplot(4, 5, i + 1)
            plt.plot(force_est[i, :])
            plt.plot(force_ref[i, 0:Ns:rt_ratio], c="red")
            plt.title(biorbd_model.muscleNames()[i].to_string())
        plt.legend(labels=["f_est", "f_ref"], bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
        get_markers = markers_fun(biorbd_model)
        markers = np.zeros((3, biorbd_model.nbMarkers(), q_ref.shape[1]))
        for i in range(q_ref.shape[1]):
            markers[:, :, i] = get_markers(q_ref[:, i])
        markers_est = np.zeros((3, biorbd_model.nbMarkers(), X_est.shape[1]))
        for i in range(X_est.shape[1]):
            markers_est[:, :, i] = get_markers(X_est[: biorbd_model.nbQ(), i])

        plt.figure("Markers")
        for i in range(markers_target.shape[1]):
            plt.plot(markers_target[:, i, 0:Ns:rt_ratio].T, "k")
            plt.plot(markers[:, i, 0:Ns:rt_ratio].T, "r--")
            plt.plot(markers_est[:, i, :].T, "b")
        plt.xlabel("Time")
        plt.ylabel("Markers Position")

        return plt.show()


def init_plot_emg(nb_emg, muscle_names=()):
    """
            Initialize pyqtgraph for emg
            ----------
            nb_emg: int
                Number of emg data to plot

            Returns
            ----------
            Init pyqtgraph for update function
    """
    app = pg.mkQApp()
    remote = []
    layout = pg.LayoutWidget()
    layout.resize(800, 800)
    label = QtGui.QLabel()
    box = []
    rplt = []
    row_count = 0
    col_span = 4 if nb_emg > 8 else 8
    for emg in range(nb_emg):
        remote.append(rgv.RemoteGraphicsView())
        remote[emg].pg.setConfigOptions(antialias=True)
        app.aboutToQuit.connect(remote[emg].close)
        if len(muscle_names) == 0:
            box.append(QtGui.QCheckBox(f"muscle_{emg}"))
        else:
            box.append(QtGui.QCheckBox(muscle_names[emg]))
        if emg >= 8:
            layout.addWidget(box[emg], row=1, col=emg - 8)
            layout.addWidget(remote[emg], row=emg - 8 + 2, col=4, colspan=col_span)
        else:
            layout.addWidget(box[emg], row=0, col=emg)
            layout.addWidget(remote[emg], row=emg + 2, col=0, colspan=col_span)
        rplt.append(remote[emg].pg.PlotItem())
        rplt[emg]._setProxyOptions(deferGetattr=True)  ## speeds up access to rplt.plot
        remote[emg].setCentralItem(rplt[emg])
        layout.addWidget(label)
        layout.show()
        emg += 1
        row_count += 1

    return rplt, layout, app, box


def update_plot_emg(emg_data, rplt, app, box):
    """
            update EMG plot
            ----------
            emg: np.ndarray
                array of muscle size
            rplt, app, box:
                values from init function
            Returns
            ----------
    """
    for emg in range(emg_data.shape[0]):
        if box[emg].isChecked() is True:
            rplt[emg].plot(emg_data[emg, :], clear=True, _callSync="off")
    app.processEvents()

    return app


def init_plot_q(nb_q, dof_names=None):
    """
        Initialize pyqtgraph for state
        ----------
        nb_q: int
            number of degree of freedom

        Returns
        ----------
        Init pyqtgraph for update function
    """
    app_q = pg.mkQApp("q")
    layout = pg.LayoutWidget()
    layout.resize(900, 800)
    layout.move(500, 0)
    label = QtGui.QLabel()
    box = []
    remote = []
    rplt = []
    row_count = 0
    col_count = 0
    # col_span = 4 if nb_q > 8 else 8
    for q in range(nb_q):
        remote.append(rgv.RemoteGraphicsView())
        remote[q].pg.setConfigOptions(antialias=True)
        app_q.aboutToQuit.connect(remote[q].close)
        names = dof_names[q] if dof_names else f"angle_{q}"
        box.append(QtGui.QCheckBox(names))
        box[q].setChecked(True)
        layout.addWidget(box[q], row=0, col=q)
        layout.addWidget(remote[q], row=row_count + 1, col=col_count)
        rplt.append(remote[q].pg.PlotItem())
        rplt[q]._setProxyOptions(deferGetattr=True)  ## speeds up access to rplt.plot
        remote[q].setCentralItem(rplt[q])
        layout.addWidget(label)
        layout.show()
        if col_count < 4:
            col_count += 1
        else:
            col_count = 0
            row_count += 1

    return rplt, layout, app_q, box


def init_plot_force(nb_mus, plot_type="progress_bar"):
    """
    Initialize pyqtgraph for force data
    ----------
    nb_mus: int
        number of muscle

    Returns
    ----------
        Init pyqtgraph for update function
    """

    if plot_type == 'curve':
        # --- Curve graph --- #
        app = pg.mkQApp('Muscle forces')
        remote = []
        layout = pg.LayoutWidget()
        layout.resize(800, 800)
        label = QtGui.QLabel()
        box = []
        rplt = []
        row_count = 0
        col_span = 4 if nb_mus > 8 else 8
        for mus in range(nb_mus):
            remote.append(rgv.RemoteGraphicsView())
            remote[mus].pg.setConfigOptions(antialias=True)
            app.aboutToQuit.connect(remote[mus].close)
            box.append(QtGui.QCheckBox(f"muscle_{mus}"))
            if mus >= 8:
                layout.addWidget(box[mus], row=1, col=mus-8)
                layout.addWidget(remote[mus], row=mus - 8 + 2, col=4, colspan=col_span)
            else:
                layout.addWidget(box[mus], row=0, col=mus)
                layout.addWidget(remote[mus], row=mus + 2, col=0, colspan=col_span)
            rplt.append(remote[mus].pg.PlotItem())
            rplt[mus]._setProxyOptions(deferGetattr=True)  ## speeds up access to rplt.plot
            remote[mus].setCentralItem(rplt[mus])
            layout.addWidget(label)
            layout.show()
            row_count += 1
        return rplt, layout, app, box

    elif plot_type == "progress_bar":
        # --- Progress bar graph --- #
        app = pg.mkQApp('Muscle forces')
        layout = pg.LayoutWidget()
        layout.resize(400, 800)
        layout.move(0, 0)
        rplt = []
        row_count = 0
        for mus in range(nb_mus):
            rplt.append(QProgressBar())
            rplt[mus].setMaximum(1000)
            layout.addWidget(rplt[mus], row=mus, col=0)
            layout.show()
            row_count += 1

        return rplt, layout, app

    elif plot_type == "bar":
        # --- Bar graph --- #
        app = pg.mkQApp('Muscle forces')
        layout = pg.plot()
        layout.resize(800, 800)
        rplt = pg.BarGraphItem(x=range(nb_mus), height=np.zeros((nb_mus)), width=0.3, brush='r')
        layout.addItem(rplt)
        return rplt, layout, app

    else:
        raise RuntimeError("Plot type not allowed")

def update_plot_force(force_est, rplt, app, ratio, muscle_names=None):  # , box):
    """
            update force plot
            ----------
            force_est: np.ndarray
                array of force estimate size
            rplt, app, box:
                values from init function
            Returns
            ----------
    """
    # --- curve --- #
    # for force in range(force_est.shape[0]):
    #     if box[force].isChecked() is True:
    #         rplt[force].plot(force_est[force, :], clear=True, _callSync='off')
    # app.processEvents()

    # --- progress bar --- #
    for force in range(force_est.shape[0]):
        value = np.mean(force_est[force, -ratio:])
        rplt[force].setValue(int(value))
        names = muscle_names[force] if muscle_names else f"muscle_{force}"
        rplt[force].setFormat(f"{names}: {int(value)} N")
    app.processEvents()

    # --- bar --- #
    # y = []
    # for force in range(force_est.shape[0]):
    #     y.append(np.mean(force_est[force, -ratio:]))
    # rplt.setOpts(height=y)
    # app.processEvents()


def update_plot_q(q_est, rplt, app, box):
    """
        update force plot
        ----------
        q_est: np.ndarray
            array of state size
        rplt, app, box:
            values from init function
        Returns
        ----------
    """
    for q in range(q_est.shape[0]):
        if box[q].isChecked() is True:
            rplt[q].plot(q_est[q, :], clear=True, _callSync="off")
    app.processEvents()
