"""
This file is part of biosiglive. It contains biorbd specific functions for musculoskeletal analysis.
"""
try:
    import biorbd
except ModuleNotFoundError:
    pass
import numpy as np


def compute_inverse_kinematics(markers, model, return_q_dot=True, kalman=None, use_kalman=True):
    """
    Function to apply the Kalman filter to the markers.
    Parameters
    ----------
    markers : numpy.array
        The experimental markers.
    model : biorbd.Model
        The model used to compute the kinematics.
    return_q_dot : bool
        If True, the function will return the q_dot.
    kalman : biorbd.Kalman
        The Kalman filter to use.
    use_kalman : bool
        If True, the function will use the Kalman filter. If False, it will use the standard inverse kinematics.

    Returns
    -------
    numpy.array or tuple
        The joint angle and (if asked) velocity. .
    """
    if use_kalman:
        markers_over_frames = []
        if not kalman:
            freq = 100  # Hz
            params = biorbd.KalmanParam(freq)
            kalman = biorbd.KalmanReconsMarkers(model, params)

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
    else:
        ik = biorbd.InverseKinematics(model, markers)
        ik.solve("only_lm")
        q_recons = ik.q
        q_dot_recons = np.array([0] * ik.nb_q)[:, np.newaxis]
    # compute markers from
    if return_q_dot:
        return q_recons, q_dot_recons
    else:
        return q_recons


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
