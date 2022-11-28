"""
This file is part of biosiglive. It contains biorbd specific functions for musculoskeletal analysis.
"""
try:
    import biorbd

    biordb_package = True
except ModuleNotFoundError:
    biordb_package = False
import numpy as np
from ..enums import InverseKinematicsMethods
from typing import Union
import time


class MskFunctions:
    def __init__(self, model: str, data_buffer_size: int = 1):
        """
        Initialize the MskFunctions class.
        Parameters
        ----------
        model : str
            Path to the biorbd model used to compute the kinematics.
        """
        if not biordb_package:
            raise ModuleNotFoundError("Biorbd is not installed. Please install it to use this function.")
        if isinstance(model, str):
            self.model = biorbd.Model(model)
        else:
            self.model = model
        self.process_time = []
        self.markers_buffer = []
        self.kin_buffer = []
        self.data_windows = data_buffer_size
        self.kalman = None

    def compute_inverse_kinematics(
        self,
        markers: np.ndarray,
        method: Union[InverseKinematicsMethods, str] = InverseKinematicsMethods.BiorbdLeastSquare,
        kalman_freq: Union[int, float] = 100,
        kalman: callable = None,
        custom_function: callable = None,
        **kwargs,
    ) -> tuple:
        """
        Function to apply the Kalman filter to the markers.
        Parameters
        ----------
        markers : numpy.array
            The experimental markers.
        kalman : biorbd.KalmanReconsMarkers
            The Kalman filter to use.
        kalman_freq : int
            The frequency of the Kalman filter.
        method : Union[InverseKinematicsMethods, str]
            The method to use to compute the inverse kinematics.
        custom_function : callable
            Custom function to use.
        Returns
        -------
        tuple
            The joint angle and velocity.
        """
        tic = time.time()
        if isinstance(method, str):
            if method in [t.value for t in InverseKinematicsMethods]:
                method = InverseKinematicsMethods(method)
            else:
                raise ValueError(f"Method {method} is not supported")

        if method == InverseKinematicsMethods.BiorbdKalman:
            self.kalman = kalman if kalman else self.kalman
            if not kalman and not self.kalman:
                freq = kalman_freq  # Hz
                params = biorbd.KalmanParam(freq)
                self.kalman = biorbd.KalmanReconsMarkers(self.model, params)
            markers_over_frames = []
            q = biorbd.GeneralizedCoordinates(self.model)
            q_dot = biorbd.GeneralizedVelocity(self.model)
            qd_dot = biorbd.GeneralizedAcceleration(self.model)
            for i in range(markers.shape[2]):
                markers_over_frames.append([biorbd.NodeSegment(m) for m in markers[:, :, i].T])

            q_recons = np.zeros((self.model.nbQ(), len(markers_over_frames)))
            q_dot_recons = np.zeros((self.model.nbQ(), len(markers_over_frames)))

            for i, targetMarkers in enumerate(markers_over_frames):
                self.kalman.reconstructFrame(self.model, targetMarkers, q, q_dot, qd_dot)
                q_recons[:, i] = q.to_array()
                q_dot_recons[:, i] = q_dot.to_array()

        elif method == InverseKinematicsMethods.BiorbdLeastSquare:
            ik = biorbd.InverseKinematics(self.model, markers)
            ik.solve("only_lm")
            q_recons = ik.q
            q_dot_recons = np.array([0] * ik.nb_q)[:, np.newaxis]

        elif method == InverseKinematicsMethods.Custom:
            if not custom_function:
                raise ValueError("No custom function provided.")
            q_recons = custom_function(markers, **kwargs)
            q_dot_recons = np.zerros((q_recons.shape()))
        else:
            raise ValueError(f"Method {method} is not supported")
        if len(self.kin_buffer) == 0:
            self.kin_buffer = [q_recons, q_dot_recons]
        else:
            self.kin_buffer[0] = np.append(self.kin_buffer[0], q_recons, axis=1)
            self.kin_buffer[1] = np.append(self.kin_buffer[1], q_dot_recons, axis=1)
        for i in range(len(self.kin_buffer)):
            self.kin_buffer[i] = self.kin_buffer[i][:, -self.data_windows :]
        self.process_time.append(time.time() - tic)
        return self.kin_buffer[0], self.kin_buffer[1]

    def compute_direct_kinematics(self, states: np.ndarray) -> np.ndarray:
        """
        Compute the direct kinematics.
        Parameters
        ----------
        states : np.ndarray
            The states to compute the direct kinematics.
        Returns
        -------
        np.ndarray
            The markers.
        """
        tic = time.time()
        if not biordb_package:
            raise ModuleNotFoundError("Biorbd is not installed. Please install it to use this function.")
        if isinstance(states, list):
            states = np.array(states)
        if states.shape[0] != self.model.nbQ():
            raise ValueError(f"States must have {self.model.nbQ()} rows.")
        if len(states.shape) != 2:
            states = states[:, np.newaxis]

        markers = np.zeros((3, self.model.nbMarkers(), states.shape[1]))
        for i in range(states.shape[1]):
            markers[:, :, i] = np.array([mark.to_array() for mark in self.model.markers(states[:, i])]).T
        if len(self.markers_buffer) == 0:
            self.markers_buffer = markers
        else:
            self.markers_buffer = np.append(self.markers_buffer, markers, axis=2)
        self.markers_buffer = self.markers_buffer[:, :, -self.data_windows :]
        self.process_time.append(time.time() - tic)
        return self.markers_buffer

    def get_mean_process_time(self):
        """
        Get the mean process time.
        Returns
        -------
        float
            The mean process time.
        """
        return np.mean(self.process_time)
