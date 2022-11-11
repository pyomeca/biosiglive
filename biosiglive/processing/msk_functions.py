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


def compute_inverse_kinematics(
    markers: np.ndarray,
    model: callable,
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
    model : biorbd.Model
        The model used to compute the kinematics.
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
    if not biordb_package:
        raise ModuleNotFoundError("Biorbd is not installed. Please install it to use this function.")
    if isinstance(method, str):
        if method in [t.value for t in InverseKinematicsMethods]:
            method = InverseKinematicsMethods(method)
        else:
            raise ValueError(f"Method {method} is not supported")

    if method == InverseKinematicsMethods.BiorbdKalman:
        if not kalman:
            freq = kalman_freq  # Hz
            params = biorbd.KalmanParam(freq)
            kalman = biorbd.KalmanReconsMarkers(model, params)
        markers_over_frames = []
        q = biorbd.GeneralizedCoordinates(model)
        q_dot = biorbd.GeneralizedVelocity(model)
        qd_dot = biorbd.GeneralizedAcceleration(model)
        for i in range(markers.shape[2]):
            markers_over_frames.append([biorbd.NodeSegment(m) for m in markers[:, :, i].T])

        q_recons = np.zeros((model.nbQ(), len(markers_over_frames)))
        q_dot_recons = np.zeros((model.nbQ(), len(markers_over_frames)))

        for i, targetMarkers in enumerate(markers_over_frames):
            kalman.reconstructFrame(model, targetMarkers, q, q_dot, qd_dot)
            q_recons[:, i] = q.to_array()
            q_dot_recons[:, i] = q_dot.to_array()

    elif method == InverseKinematicsMethods.BiorbdLeastSquare:
        ik = biorbd.InverseKinematics(model, markers)
        ik.solve("only_lm")
        q_recons = ik.q
        q_dot_recons = np.array([0] * ik.nb_q)[:, np.newaxis]

    elif method == InverseKinematicsMethods.Custom:
        if not custom_function:
            raise ValueError("No custom function provided.")
        q_recons = custom_function(markers, **kwargs)
        q_dot_recons = np.zerros((q_recons.shape()))
    # compute markers from
    return q_recons, q_dot_recons


# def markers_fun(biorbd_model, q=None, eigen_backend=False):
#     if eigen_backend:
#         return [biorbd_model.markers(q)[i].to_array() for i in range(biorbd_model.nbMarkers())]
#     else:
#         qMX = MX.sym("qMX", biorbd_model.nbQ())
#         return Function(
#             "markers",
#             [qMX],
#             [horzcat(*[biorbd_model.markers(qMX)[i].to_mx() for i in range(biorbd_model.nbMarkers())])],
#         )
