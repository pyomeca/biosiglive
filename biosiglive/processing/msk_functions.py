"""
This file contains biorbd specific functions for musculoskeletal analysis such as inverse or direct kinematics.
"""

try:
    import biorbd

    biordb_package = True
except ModuleNotFoundError:
    biordb_package = False
import numpy as np
from ..enums import InverseKinematicsMethods
from .data_processing import RealTimeProcessing
from typing import Union
import time

try:
    import casadi as ca
    from acados_template import AcadosOcp, AcadosModel, AcadosOcpSolver
    from .msk_utils import (
        _init_acados,
        _update_solver,
        _init_casadi_function,
        _create_new_model,
        _compute_forces,
        _compute_inverse_dynamics,
        _express_in_new_coordinate,
        ExternalLoads,
    )
except ModuleNotFoundError:
    pass


class MskFunctions:
    def __init__(self, model: str, data_buffer_size: int = 1, system_rate: int = 100):
        """
        The MskFunctions contains all function for some musculoskeletal methods.

        Parameters
        ----------
        model : Union[str, biorbd.Model]
            Path to the biorbd model used to compute the kinematics.
        data_buffer_size: int
            The size of the buffer used to store the data.
        system_rate: int
            The working frequency of the input data (markers or joint kinematics).
        """
        self.weigh_list = None
        self.model_all_dofs = None
        self.ca_model = None
        self.once_compile = None
        self.ocp = None
        self.mjt_funct = None
        self.ocp_solver = None
        if not biordb_package:
            raise ModuleNotFoundError(
                "Biorbd is not installed."
                " Please install it via"
                " 'conda install biorbd -cconda-forge' to use this function."
            )
        if isinstance(model, str):
            self.model = biorbd.Model(model)
        else:
            self.model = model
        self.process_time = []
        self.markers_buffer = []
        self.kin_buffer = []
        self.dyn_buffer = []
        self.act_buffer = []
        self.id_state_buffer = []
        self.tau_buffer = []
        self.jrf_buffer = []
        self.res_tau_buffer = []
        self.data_windows = data_buffer_size
        self.system_rate = system_rate
        self.kalman = None
        self._rt_process_methods = None
        self._state_idx_to_process = None
        self.q_mapping, self.ordered_seg, self.ordered_idx = None, None, None

    def clean_all_buffers(self):
        for key in self.__dict__.keys():
            self.__dict__[key] = [] if "buffer" in key else self.__dict__[key]

    def compute_inverse_kinematics(
        self,
        markers: np.ndarray,
        method: Union[InverseKinematicsMethods, str] = InverseKinematicsMethods.BiorbdLeastSquare,
        kalman_freq: Union[int, float] = 100,
        kalman: callable = None,
        custom_function: callable = None,
        initial_guess: Union[np.ndarray, list] = None,
        qdot_from_finite_difference: bool = False,
        noise_factor=1e-10,
        error_factor=1e-5,
        **kwargs,
    ) -> tuple:
        """
        Function to apply the inverse kinematics using the markers data and a biorbd model type.

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
        initial_guess: Union[np.ndarray, list]
            Initial generalized coordinate, velocity and acceleration for the kalman filter
        qdot_from_finite_difference: bool
            If true the velocity will be computed using a finite difference method.

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
        if qdot_from_finite_difference and markers.shape[2] < 2:
            raise ValueError("You must have at least two frames to compute the velocity using finite difference.")
        if method == InverseKinematicsMethods.BiorbdKalman:
            self.kalman = kalman if kalman else self.kalman
            if not kalman and not self.kalman:
                freq = kalman_freq  # Hz
                params = biorbd.KalmanParam(freq, noiseFactor=noise_factor, errorFactor=error_factor)
                self.kalman = biorbd.KalmanReconsMarkers(self.model, params)
            if initial_guess:
                if isinstance(initial_guess, np.ndarray):
                    if initial_guess.shape[0] != 3:
                        raise RuntimeError("Initial guess must have dims : 3xNdofxNframes if give as an array.")
                    initial_guess = [initial_guess[0, ...], initial_guess[1, ...], initial_guess[2, ...]]
                for i in initial_guess:
                    if len(i.shape) != 1:
                        raise RuntimeError("initial guess must be 1D array.")
                    if i.shape[0] != self.model.nbQ():
                        raise RuntimeError("Inital guess msut have the same size than model DOfs.")
                if len(initial_guess) != 3:
                    raise RuntimeError("Initial guess must be of len 3 (angle, velocity, acceleration).")
                self.kalman.setInitState(initial_guess[0], initial_guess[1], initial_guess[2])

            q = biorbd.GeneralizedCoordinates(self.model)
            q_dot = biorbd.GeneralizedVelocity(self.model)
            qd_dot = biorbd.GeneralizedAcceleration(self.model)
            q_recons = np.zeros((self.model.nbQ(), markers.shape[2]))
            q_dot_recons = np.zeros((self.model.nbQ(), markers.shape[2]))
            q_ddot_recons = np.zeros((self.model.nbQ(), markers.shape[2]))
            for i in range(markers.shape[2]):
                self.kalman.reconstructFrame(
                    self.model, [biorbd.NodeSegment(m) for m in markers[:, :, i].T], q, q_dot, qd_dot
                )
                q_recons[:, i] = q.to_array()
                q_dot_recons[:, i] = q_dot.to_array()
                q_ddot_recons[:, i] = qd_dot.to_array()
        elif method == InverseKinematicsMethods.BiorbdLeastSquare:
            ik = biorbd.InverseKinematics(self.model, markers)
            ik.solve("only_lm")
            # ik.solve("trf")

            q_recons = ik.q
            q_dot_recons = np.array([0] * ik.nb_q)[:, np.newaxis]
            q_ddot_recons = np.array([0] * ik.nb_q)[:, np.newaxis]

        elif method == InverseKinematicsMethods.Custom:
            if not custom_function:
                raise ValueError("No custom function provided.")
            q_recons = custom_function(markers, **kwargs)
            q_dot_recons = np.zeros((q_recons.shape()))
            q_ddot_recons = np.zeros((q_recons.shape()))
        else:
            raise ValueError(f"Method {method} is not supported")

        if qdot_from_finite_difference:
            q_dot_recons = np.zeros_like(q_recons)
            for i in range(1, q_recons.shape[1] - 1):
                q_dot_recons[:, i] = (q_recons[:, i + 1] - q_recons[:, i - 1]) / (2 / self.system_rate)

        if len(self.kin_buffer) == 0:
            self.kin_buffer = [q_recons, q_dot_recons, q_ddot_recons]
        else:
            self.kin_buffer[0] = np.append(self.kin_buffer[0], q_recons, axis=1)
            self.kin_buffer[1] = np.append(self.kin_buffer[1], q_dot_recons, axis=1)
            self.kin_buffer[2] = np.append(self.kin_buffer[2], q_ddot_recons, axis=1)
        for i in range(len(self.kin_buffer)):
            self.kin_buffer[i] = self.kin_buffer[i][:, -self.data_windows :]
        self.process_time.append(time.time() - tic)
        return self.kin_buffer[0].copy(), self.kin_buffer[1].copy(), self.kin_buffer[2].copy()

    def compute_direct_kinematics(self, states: np.ndarray) -> np.ndarray:
        """
        Compute the direct kinematics using the joint angle and a biorbd model type.

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
            raise ModuleNotFoundError(
                "Biorbd is not installed."
                " Please install it via"
                " 'conda install biorbd -cconda-forge' to use this function."
            )
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

    def compute_inverse_dynamics(
        self,
        joint_positions: np.ndarray = None,
        joint_velocities: np.ndarray = None,
        joint_accelerations: np.ndarray = None,
        state_idx_to_process: list = (),
        lowpass_frequency: Union[list, int] = None,
        windows_length: Union[list, int] = None,
        positions_from_inverse_kinematics: bool = False,
        velocities_from_inverse_kinematics: bool = False,
        accelerations_from_inverse_kinematics: bool = False,
        external_load: any = None,
    ) -> np.ndarray:
        """
        Compute the inverse dynamics using the model kinematics and a biorbd model type.

        Parameters
        ----------
        joint_positions : np.ndarray
            The joint position for each model joints
        joint_velocities : np.ndarray
            The joint velocities for each model joints
        joint_accelerations : np.ndarray
            The joint accelerations for each model joints
        state_idx_to_process: list
            The list of the index of the data to apply a low pass filter on. If empty no filter will be applied.
        lowpass_frequency: Union[list, int]
            the list of the frequency for the low pass filter, in the same order as the index, if an integer is given
            the same value will be used for each index
        windows_length: Union[list, int]
            The list of the window length for the moving average filter,
             in the same order as the index, if an integer is given the same value will be used for each index
        positions_from_inverse_kinematics: bool
            If true use the result of precomputed inverse kinematics.
             Note that the inverse kinematics must be computed before.
        velocities_from_inverse_kinematics: bool
            If true use the result of precomputed inverse kinematics.
             Note that the inverse kinematics must be computed before.
        Returns
        -------
        np.ndarray
            The generalized torque.
        """
        tic = time.time()
        if not biordb_package:
            raise ModuleNotFoundError(
                "Biorbd is not installed."
                " Please install it via"
                " 'conda install biorbd -cconda-forge' to use this function."
            )
        if (positions_from_inverse_kinematics or velocities_from_inverse_kinematics) and len(self.kin_buffer) == 0:
            raise ValueError("Inverse kinematics must be called before using kinematics results.")
        states_init = [joint_positions, joint_velocities, joint_accelerations]
        if isinstance(joint_positions, np.ndarray):
            if len(joint_positions.shape) > 1 and joint_positions.shape[1] > 1:
                raise RuntimeError(
                    "Data must be only for one frame," " please do a for loop if you need ID for more than one frame. "
                )
        if positions_from_inverse_kinematics:
            states_init[0] = self.kin_buffer[0][:, -1:]
        if velocities_from_inverse_kinematics:
            states_init[1] = self.kin_buffer[1][:, -1:]
        if accelerations_from_inverse_kinematics:
            states_init[2] = self.kin_buffer[2][:, -1:]
        if not positions_from_inverse_kinematics and not joint_positions:
            raise RuntimeError("Please provide at lease the joint position to compute the inverse dynamics.")
        has_changed = self._state_idx_to_process != state_idx_to_process
        self._state_idx_to_process = state_idx_to_process
        if len(state_idx_to_process) != 0:
            states_init = self._filter_states(
                states_init, state_idx_to_process, windows_length, lowpass_frequency, has_changed
            )

        states = self._check_states(states_init)
        tau = np.zeros((self.model.nbQ(), 1))
        # for i in range(tau.shape[1]):
        if external_load is not None:
            external_biorbd_loads = external_load.to_biorbd_loads(self.model)
            tau[:, 0] = self.model.InverseDynamics(
                states[0][:, -1], states[1][:, -1], states[2][:, -1], external_biorbd_loads
            ).to_array()
        else:
            tau[:, 0] = self.model.InverseDynamics(states[0][:, -1], states[1][:, -1], states[2][:, -1]).to_array()
        self.tau_buffer = (
            tau if len(self.tau_buffer) == 0 else np.append(self.tau_buffer[:, -self.data_windows + 1 :], tau, axis=1)
        )
        self.process_time.append(time.time() - tic)
        return self.tau_buffer.copy()

    def compute_static_optimization(
        self,
        q: np.ndarray = None,
        q_dot: np.ndarray = None,
        tau: np.ndarray = None,
        ocp_solver: any = None,
        compile_c_code: bool = True,
        use_residual_torque: bool = True,
        torque_tracking_as_objective: bool = True,
        muscle_torque_dynamics_func: any = None,
        scaling_factor: Union[list, tuple] = (1, 1),
        muscle_track_idx: list = None,
        emg: np.ndarray = None,
        weight: dict = None,
        x0: np.ndarray = None,
        data_from_inverse_dynamics: bool = False,
        solver_options: dict = None,
        compile_only_first_call: bool = True,
        print_optimization_status: bool = False,
    ):
        """
        Compute the static optimization using the model kinematics and a biorbd model type.
        Parameters
        ----------
        q: np.ndarray
            The joint position for each model joints
        q_dot: np.ndarray
            The joint velocities for each model joints
        tau: np.ndarray
            The joint torques for each model joints
        ocp_solver: AcadosOcpSolver
            The acados solver to use, if none is provided a new one will be created
        compile_c_code: bool
            If true the c code will be compiled
        use_residual_torque: bool
            If true the residual torque will be used
        torque_tracking_as_objective: bool
            If true the torque tracking will be used as objective, otherwise it will be a constraint
        muscle_torque_dynamics_func: ca.Function
            The casadi function to use for the muscle torque dynamics (muscle activation -> muscle torque)
        scaling_factor: list
            The scaling factor to use for the muscle activation and muscle torque optimization variables
        muscle_track_idx: list
            The index of the muscles to track
        emg: np.ndarray
            The emg data to track for the muscle_track_idx provided
        weight: dict
            The weight to use for the objective function
        x0: np.ndarray
            The initial guess for the optimization
        data_from_inverse_dynamics: bool
            If true the data will be taken from the inverse dynamics (q, qdot, tau) instead of the provided data
        solver_options: dict
            The solver options to use for the acados solver
        compile_only_first_call: bool
            If true the c code will be compiled only for the first call
        print_optimization_status: bool
            If true the status of the optimization will be printed
        Returns
        -------
        tuple:
            The optimal activation and torque for each muscles
        """
        if muscle_track_idx and emg is None:
            muscle_track_idx = None
        if emg is not None and not muscle_track_idx:
            raise RuntimeError("If you want to track muscles, you must provide the muscle index to track.")

        if not self.ca_model:
            import biorbd_casadi as biorbd_ca

            self.ca_model = biorbd_ca.Model(self.model.path().absolutePath().to_string())
        if data_from_inverse_dynamics:
            if len(self.id_state_buffer) == 0 or len(self.tau_buffer) == 0:
                raise RuntimeError("You must have called the inverse dynamics before using the data from it.")
            q = self.id_state_buffer[0][:, -1:]
            q_dot = self.id_state_buffer[1][:, -1:]
            tau = self.tau_buffer[:, -1:]
        if q is None or q_dot is None or tau is None:
            raise RuntimeError(
                "Please provide q, q_dot and tau to compute the static optimization."
                " Or use data from inverse dynamics."
            )
        if (
            q.shape[0] != self.ca_model.nbQ()
            or q_dot.shape[0] != self.ca_model.nbQ()
            or tau.shape[0] != self.ca_model.nbQ()
        ):
            raise RuntimeError("The provided data must have the same number of dof as the model.")
        if isinstance(q, np.ndarray):
            if len(q.shape) > 1 and q.shape[1] > 1:
                raise RuntimeError(
                    "Data must be only for one frame," " please do a for loop if you need ID for more than one frame. "
                )
        if isinstance(q_dot, np.ndarray):
            if len(q_dot.shape) > 1 and q_dot.shape[1] > 1:
                raise RuntimeError(
                    "Data must be only for one frame," " please do a for loop if you need ID for more than one frame. "
                )
        if isinstance(tau, np.ndarray):
            if len(tau.shape) > 1 and tau.shape[1] > 1:
                raise RuntimeError(
                    "Data must be only for one frame," " please do a for loop if you need ID for more than one frame. "
                )

        self.ocp_solver = ocp_solver if ocp_solver else self.ocp_solver
        self.mjt_funct = muscle_torque_dynamics_func if muscle_torque_dynamics_func else self.mjt_funct
        self.mjt_funct = self.mjt_funct if self.mjt_funct else _init_casadi_function(self.ca_model)
        if not compile_c_code and not self.ocp_solver:
            raise RuntimeError("You must provide a solver if you want avoid to compile c code.")
        compile_c_code = not self.once_compile if compile_only_first_call else compile_c_code
        if not self.ocp_solver or compile_c_code:
            if not self.ocp:
                self.ocp, self.weigh_list = _init_acados(
                    self.ca_model,
                    torque_tracking_as_objective,
                    self.mjt_funct,
                    use_residual_torque,
                    scaling_factor,
                    muscle_track_idx,
                    weight,
                    solver_options,
                    emg=emg,
                )

            self.ocp_solver = AcadosOcpSolver(
                self.ocp, json_file=f"{self.ocp.model.name}.json", build=compile_c_code, generate=True
            )
            self.once_compile = True

        target = np.zeros((len(self.weigh_list)))
        # if emg is not None:
        #     target[np.where(np.array(self.weigh_list) == "tracking_emg")] = emg[:, 0]
        self.ocp_solver = _update_solver(
            self.ocp_solver, target, x0, q, q_dot, tau, emg=emg, torque_as_objective=torque_tracking_as_objective
        )

        self.ocp_solver.solve()
        solution = self.ocp_solver.get(0, "x")
        if print_optimization_status:
            print("---------- QP Solver statistics ----------")
            self.ocp_solver.print_statistics()
            print("Residuals: ", self.ocp_solver.get_stats("residuals"))
            print("Optimization status: ", self.ocp_solver.status, "\n")
        muscle_activations = solution[: self.ca_model.nbMuscles() * q.shape[1]] / scaling_factor[0]
        residual_torque = solution[self.ca_model.nbMuscles() * q.shape[1] :] / scaling_factor[1]
        self.act_buffer = (
            np.append(self.act_buffer[:, -self.data_windows + 1 :], muscle_activations[:, np.newaxis], axis=1)
            if len(self.act_buffer) != 0
            else muscle_activations[:, np.newaxis]
        )
        self.res_tau_buffer = (
            np.append(self.res_tau_buffer[:, -self.data_windows + 1 :], residual_torque[:, np.newaxis], axis=1)
            if len(self.res_tau_buffer) != 0
            else residual_torque[:, np.newaxis]
        )
        return self.act_buffer.copy(), self.res_tau_buffer.copy()

    def compute_joint_reaction_load(
        self,
        q: np.ndarray = None,
        qdot: np.ndarray = None,
        qddot: np.ndarray = None,
        muscle_activations: np.ndarray = None,
        # residual_torques: np.ndarray = None,
        act_from_static_optimisation: bool = False,
        kinetics_from_inverse_dynamics: bool = False,
        express_in_coordinate: str = None,
        apply_on_segment: str = "all",
        from_distal: bool = True,
        application_point: Union[list, tuple] = None,
        external_loads: any = None,
    ):
        if act_from_static_optimisation and len(self.act_buffer) == 0:
            raise RuntimeError("You must compute muscle activation from static optimisation before using them.")
        if (act_from_static_optimisation and muscle_activations is not None) or (
            not act_from_static_optimisation and muscle_activations is None
        ):
            raise RuntimeError(
                "Please provide one muscle activation source. Either from static optimisation or " "from the user."
            )
        if kinetics_from_inverse_dynamics and len(self.tau_buffer) == 0:
            raise RuntimeError("You must compute joint kinetics from inverse dynamics before using them.")
        if (kinetics_from_inverse_dynamics and np.sum((q, qdot, qddot)) is not None) or (
            not kinetics_from_inverse_dynamics and np.sum((q, qdot, qddot)) is None
        ):
            raise RuntimeError("Please provide one kinetics source. Either from inverse dynamics or " "from the user.")
        add_idx = 0 if not from_distal else 1
        if self.model.segments()[-1].name().to_string() in apply_on_segment and from_distal:
            raise RuntimeError(
                "Can not give force from distal segment on the last segment."
                "Please consider using directly the inverse dynamics."
            )
        non_virtual_segments = [
            seg.name().to_string() for seg in self.model.segments() if seg.characteristics().mass() > 1e-7
        ]

        final_target_segments = (
            [
                non_virtual_segments[i + add_idx]
                for i in range(len(non_virtual_segments) - 1)
                if non_virtual_segments[i] in apply_on_segment
            ]
            if apply_on_segment != "all"
            else non_virtual_segments
        )
        if apply_on_segment != "all" and non_virtual_segments[-1] in apply_on_segment:
            final_target_segments.append(non_virtual_segments[-1])
        express_in_coordinate = (
            [express_in_coordinate] if isinstance(express_in_coordinate, str) else express_in_coordinate
        )
        application_point = [application_point] if not isinstance(application_point, list) else application_point
        application_point = [[0, 0, 0]] * len(final_target_segments) if not application_point else application_point
        if len(express_in_coordinate) != len(final_target_segments):
            for coord in express_in_coordinate:
                if coord not in final_target_segments:
                    raise RuntimeError(
                        "The segment provided is not a real segment but a virtual one. "
                        "Please provide a real segment to compute joint load."
                    )
            raise RuntimeError("You must provide the coordinate system to express your joint loads.")
        if len(application_point) != len(final_target_segments):
            raise RuntimeError("You must provide an application point for each wanted joint load.")
        if isinstance(q, list):
            q = np.array(q)
        if isinstance(qdot, list):
            qdot = np.array(qdot)
        if isinstance(qddot, list):
            qddot = np.array(qddot)
        if len(q.shape) != 2:
            q = q[:, np.newaxis]
        if len(qdot.shape) != 2:
            qdot = qdot[:, np.newaxis]
        if len(qddot.shape) != 2:
            qddot = qddot[:, np.newaxis]
        if len(muscle_activations.shape) != 2:
            muscle_activations = muscle_activations[:, np.newaxis]
        if not self.model_all_dofs:
            self.model_all_dofs, self.q_mapping, self.ordered_seg, self.ordered_idx = _create_new_model(
                self.model,
                final_target_segments,
            )

        q_tot = np.zeros((len(sum(self.ordered_idx, [])) + len(self.q_mapping), q.shape[1]))
        q_dot_tot = np.zeros_like(q_tot)
        q_ddot_tot = np.zeros_like(q_tot)
        q_tot[self.q_mapping], q_dot_tot[self.q_mapping], q_ddot_tot[self.q_mapping] = q, qdot, qddot
        q, qdot, qddot = q_tot, q_dot_tot, q_ddot_tot
        all_trans = np.ndarray((len(final_target_segments), 3, q.shape[1]))
        all_rot = np.ndarray((len(final_target_segments), 3, q.shape[1]))
        muscle_activations = self.act_buffer if act_from_static_optimisation else muscle_activations
        # res_torque = self.res_tau_buffer if act_from_static_optimisation else residual_torques
        # b = bioviz.Viz(loaded_model=model_all_dofs)
        # b.load_movement(q)
        # b.exec()
        for i in range(q.shape[1]):
            tic = time.time()
            all_global_jcs_old = [jcs.to_array() for jcs in self.model_all_dofs.allGlobalJCS(q[:, i])]
            inv_all_global_jcs_new = [
                np.linalg.inv(jcs.to_array()) for jcs in self.model_all_dofs.allGlobalJCS(q[:, i])
            ]
            translational_in_local, rotational_in_local = _compute_inverse_dynamics(
                self.model_all_dofs,
                q[:, i],
                qdot[:, i],
                qddot[:, i],
                segment_names=self.ordered_seg,
                segment_idx=self.ordered_idx,
                external_loads=external_loads,
            )
            if self.model_all_dofs.nbMuscles() != 0:
                trans_muscle_actions, rot_muscle_actions = _compute_forces(
                    self.model_all_dofs,
                    q[:, i],
                    qdot[:, i],
                    muscle_activations[:, i],
                    segment_names=self.ordered_seg,
                    segment_idx=self.ordered_idx,
                    compound="muscle",
                )

            if self.model.nbLigaments() != 0:
                raise RuntimeError("Ligaments are not yet implemented when computing joint loads.")
                # rot_ligament_actions, trans_ligaments_actions = _compute_forces(model_all_dofs, q[:, i], qdot[:, i],
                #                                                                   muscle_activations[:, i],
                #                                                                   segment_names=ordered_seg,
                #                                                                   segment_idx=ordered_idx,
                #                                                                 compound="ligament")
            if self.model.nbActuators() != 0:
                raise RuntimeError("Actuators are not yet implemented when computing joint loads.")
            if self.model.nbPassiveTorques() != 0:
                raise RuntimeError("Passive torques are not yet implemented when computing joint loads.")
            count = 0
            for k in range(len(self.ordered_seg)):
                if self.ordered_seg[k] in final_target_segments:
                    all_trans[count, :, i] = translational_in_local[k]
                    all_rot[count, :, i] = rotational_in_local[k]
                    if self.model_all_dofs.nbMuscles() != 0:
                        all_trans[count, :, i] = np.sum((trans_muscle_actions[k], all_trans[count, :, i]), axis=0)
                        all_rot[count, :, i] = np.sum((rot_muscle_actions[k], all_rot[count, :, i]), axis=0)
                    if express_in_coordinate:
                        segment_idx = self.model_all_dofs.getBodyBiorbdId(final_target_segments[count])
                        new_segment_idx = self.model_all_dofs.getBodyBiorbdId(express_in_coordinate[count])
                        if new_segment_idx == -1:
                            raise RuntimeError(
                                f"The segment provided ({express_in_coordinate[count]}) does not exist."
                                " Please provide a real segment."
                            )
                        all_trans[count, :, i], all_rot[count, :, i] = _express_in_new_coordinate(
                            all_trans[count, :, i],
                            all_rot[count, :, i],
                            application_point[count],
                            all_global_jcs_old[segment_idx],
                            inv_all_global_jcs_new[new_segment_idx],
                        )
                        count += 1

            # print("real_jrf_time:", time.time() - tic)
        return np.concatenate((all_trans, all_rot), axis=0)

    def _check_states(self, states):
        states_tmp = states.copy()
        for s, state in enumerate(states_tmp):
            states[s] = np.array(state) if isinstance(state, (tuple, list)) else state
            states[s] = states[s][:, np.newaxis] if states[s] is not None and len(states[s].shape) != 2 else state
        idx_to_compute_derivative = [i for i in range(len(states)) if states[i] is None]
        all_shapes = [state.shape[1] for state in states if state is not None]
        self.id_state_buffer = [None] * 3 if len(self.id_state_buffer) == 0 else self.id_state_buffer
        for i in range(len(self.id_state_buffer)):
            state_to_append = states[i] if states[i] is not None else np.zeros(states[0].shape)
            if self.id_state_buffer[i] is None:
                self.id_state_buffer[i] = state_to_append
            else:
                self.id_state_buffer[i] = np.append(
                    self.id_state_buffer[i][:, -self.data_windows + 1 :], state_to_append, axis=1
                )
        if all_shapes.count(all_shapes[0]) != len(all_shapes):
            raise RuntimeError("Buffer and given data must have the same size.")
        if len(idx_to_compute_derivative) >= 1:
            self.id_state_buffer = self._compute_differential_state(idx_to_compute_derivative)
        return self.id_state_buffer

    def _compute_differential_state(self, idx_to_compute_derivative):
        if self.data_windows <= 2:
            raise ValueError("Buffer size must be superior than 2.")
        states = np.copy(self.id_state_buffer)
        for i in range(1, len(states)):
            if i in idx_to_compute_derivative:
                derivative = (
                    (states[i - 1][:, -2:-1] - states[i - 1][:, -1:]) / (2 / self.system_rate)
                    if states[i - 1].shape[1] > 1
                    else np.zeros((states[i - 1].shape[0], 1))
                )
                self.id_state_buffer[i][:, -1:] = derivative
        return self.id_state_buffer

    def _filter_states(
        self, states, state_idx_to_process, windows_length=None, low_pass_frequency=None, has_changed=False
    ):
        self._rt_process_methods = (
            RealTimeProcessing(self.system_rate, self.data_windows)
            if not self._rt_process_methods or has_changed
            else self._rt_process_methods
        )

        states_to_process = None
        for s, state in enumerate(states):
            if s in state_idx_to_process:
                if state is None:
                    raise RuntimeError("You must provide a values before filter it.")
                states_to_process = (
                    np.append(states_to_process, state, axis=0) if states_to_process is not None else state
                )

        states_proc = self._rt_process_methods.process_emg(
            states_to_process,
            moving_average=windows_length is not None,
            low_pass_filter=low_pass_frequency is not None,
            band_pass_filter=False,
            centering=False,
            absolute_value=False,
            normalization=False,
            moving_average_window=windows_length,
            lpf_lcut=low_pass_frequency,
        )[:, -1:]
        for i, state in enumerate(states):
            states[i] = (
                states_proc[i * state.shape[0] : (i + 1) * state.shape[0], :] if i in state_idx_to_process else state
            )
        return states

    def get_mean_process_time(self):
        """
        Get the mean process time.

        Returns
        -------
        float
            The mean process time.
        """
        return np.mean(self.process_time)

    def get_kinematics_from_ik(self):
        return self.kin_buffer.copy()

    def get_tau_from_id(self):
        return self.tau_buffer.copy()

    def get_filtered_kinematics_from_id(self):
        return self.id_state_buffer.copy()

    def get_activation_from_so(self):
        return self.act_buffer.copy()

    def get_jrf_from_external_load_analysis(self):
        return self.jrf_buffer.copy()
