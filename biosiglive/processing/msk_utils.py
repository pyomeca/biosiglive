import numpy as np

try:
    import casadi as ca
except ModuleNotFoundError:
    pass
try:
    from acados_template import AcadosOcp, AcadosModel, AcadosOcpSolver

    acados_package = True
except ModuleNotFoundError:
    acados_package = False
from scipy import linalg

try:
    import biorbd
except:
    pass


def _set_solver_options(ocp, solver_options=None):
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_OSQP"  # 'PARTIAL_CONDENSING_HPIPM'  # FULL_CONDENSING_QPOASES
    # PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES, FULL_CONDENSING_HPIPM,
    # PARTIAL_CONDENSING_QPDUNES, PARTIAL_CONDENSING_OSQP
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"  # GAUSS_NEWTON, EXACT
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.sim_method_num_steps = 1
    ocp.solver_options.sim_method_num_stages = 1
    ocp.solver_options.sim_method_jac_reuse = 1
    ocp.solver_options.print_level = 0
    ocp.solver_options.tol = 1e-3
    ocp.solver_options.nlp_solver_type = "SQP_RTI"  # SQP_RTI, SQP
    # ocp.solver_options.levenberg_marquardt = 90.0
    ocp.solver_options.nlp_solver_max_iter = 1000
    ocp.solver_options.qp_solver_iter_max = 4000
    ocp.solver_options.qp_tol = 5e-5
    if solver_options:
        for key, value in solver_options.items():
            setattr(ocp.solver_options, key, value)
    return ocp


def _attribute_target_cost_and_constraints_function(model, names_J, tau, torque_tracking_as_objective):
    constr = None
    target = []
    for i in range(tau.shape[1]):
        names_J.append(["act"] * model.nbMuscles())
        target.append([None] * model.nbMuscles())
        names_J.append(["pas_tau"] * model.nbQ())
        target.append([None] * model.nbQ())
        if torque_tracking_as_objective:
            names_J.append(["tau"] * model.nbQ())
            for k in range(tau.shape[0]):
                target.append([tau[k, i]])
    target = sum(target, [])
    names_J = sum(names_J, [])
    y_ref_start = []
    for idx in range(len(names_J)):
        if target[idx] is not None:
            y_ref_start.append(target[idx])
        else:
            y_ref_start.append([0])
    return y_ref_start


def _init_acados(
    model,
    torque_tracking_as_objective,
    mjt_func,
    use_residual_torque,
    scaling_factor,
    muscle_track_idx,
    weight,
    solver_options,
    emg=None,
):
    q = ca.SX.sym("q", model.nbQ())
    qdot = ca.SX.sym("q", model.nbQ())
    tau = ca.SX.sym("tau", model.nbQ())
    emg_sym = None
    if emg is not None:
        emg_sym = ca.SX.sym("emg", emg.shape[0])
    x = ca.SX.sym("x", model.nbMuscles() * q.shape[1])
    if use_residual_torque:
        pas_tau = ca.SX.sym("pas_tau", q.shape[0])
    else:
        pas_tau = ca.SX.zeros(q.shape[0])
    lbx = (np.zeros((model.nbMuscles() * q.shape[1])) + 0.00001) * scaling_factor[0]
    ubx = np.ones((model.nbMuscles() * q.shape[1])) * scaling_factor[0]
    lh = tau[:, 0]
    uh = tau[:, 0]
    if use_residual_torque:
        lb_torque = -np.ones((q.shape[0])) * 5 * scaling_factor[1]
        ub_torque = np.ones((q.shape[0])) * 5 * scaling_factor[1]
        lb_torque[:5] = -20 * scaling_factor[1] * np.ones((5))
        ub_torque[:5] = 20 * scaling_factor[1] * np.ones((5))
        # lb_torque[-1:] = -20 * scaling_factor[1] * np.ones((1))
        # ub_torque[-1:] = 20 * scaling_factor[1] * np.ones((1))
        # lb_torque = -20 * scaling_factor[1] * np.ones((q.shape[0]))
        # ub_torque = 20 * scaling_factor[1] * np.ones((q.shape[0]))
        lbx = np.hstack((lbx, lb_torque))
        ubx = np.hstack((ubx, ub_torque))
    ocp = AcadosOcp()
    ocp.model = AcadosModel()
    ocp = _set_solver_options(ocp, solver_options)
    ocp, weigth_list = _init_cost_function(
        model,
        x,
        mjt_func,
        torque_tracking_as_objective,
        pas_tau,
        ocp,
        q,
        qdot,
        tau,
        emg_sym,
        scaling_factor,
        muscle_track_idx,
        weight,
    )
    x = ca.vertcat(x, pas_tau)
    p = ca.vertcat(q, qdot)
    if use_residual_torque:
        if torque_tracking_as_objective:
            p = ca.vertcat(p, tau)
    if emg_sym is not None:
        p = ca.vertcat(p, emg_sym)
    ocp.model.p = p
    ocp.model.disc_dyn_expr = x
    ocp.dims.np = ocp.model.p.size()[0]
    ocp.parameter_values = np.zeros((ocp.model.p.size()[0],))
    ocp.model.x = x
    ocp.dims.nx = model.nbMuscles() * q.shape[1] + q.shape[0]
    ocp.model.f_expl_expr = x
    # ocp.model.f_impl_expr = x
    ocp.model.xdot = x
    # ocp.model.con_h_expr = constr

    ocp.dims.nbx = ocp.dims.nx
    ocp.model.u = ca.SX.sym("u", 0)
    ocp.model.name = "test_acados_SO"
    T = 1
    N = 1
    ocp.dims.N = N
    ocp.solver_options.tf = T
    ocp.constraints.idxbx_0 = np.array(range(ocp.dims.nx))
    if not torque_tracking_as_objective:
        ocp.dims.nh = tau.shape[0]
        ocp.constraints.lh = np.zeros((tau.shape[0],))
        ocp.constraints.uh = np.zeros((tau.shape[0],))
    ocp.constraints.lbx_0 = lbx
    ocp.constraints.ubx_0 = ubx
    return ocp, weigth_list


def _init_cost_function(
    model,
    x,
    ca_function,
    torque_tracking_as_objective,
    pas_tau,
    ocp,
    q,
    qdot,
    tau,
    emg,
    scaling_factor,
    muscle_track_idx,
    weights,
):
    if not weights:
        weights = {"tau": 1, "act": 1, "tracking_emg": 1, "pas_tau": 1}
    # emg = np.ones((len(muscle_track_idx), q.shape[1])) * 0.7
    names_J = []
    J = None
    constr = None
    for i in range(q.shape[1]):
        mus_tau = ca_function(
            x[i * model.nbMuscles() : (i + 1) * model.nbMuscles()] / scaling_factor[0], q[:, i], qdot[:, i]
        )

        for m in range(model.nbMuscles()):
            if emg is not None and (muscle_track_idx and m in muscle_track_idx):
                idx = muscle_track_idx.index(m)
                if J is None:
                    J = (
                        (x[i * model.nbMuscles() + m : i * model.nbMuscles() + m + 1])
                        - ca.SX(emg[idx, i]) * scaling_factor[0]
                    ) ** 2
                    names_J.append(["tracking_emg"])
                    # J = ca.vertcat(J, x[i * model.nbMuscles() + m: i * model.nbMuscles() + m + 1] ** 2)
                    # names_J.append(["act"])
                else:
                    J = ca.vertcat(
                        J,
                        (
                            (x[i * model.nbMuscles() + m : i * model.nbMuscles() + m + 1])
                            - ca.SX(emg[idx, i]) * scaling_factor[0]
                        )
                        ** 2,
                    )
                    names_J.append(["tracking_emg"])
                    # J = ca.vertcat(J, x[i * model.nbMuscles() + m: i * model.nbMuscles() + m + 1] ** 2)
                    # names_J.append(["act"])
            else:
                if J is None:
                    J = x[i * model.nbMuscles() + m : i * model.nbMuscles() + m + 1] ** 2
                    names_J.append(["act"])
                else:
                    J = ca.vertcat(J, x[i * model.nbMuscles() + m : i * model.nbMuscles() + m + 1] ** 2)
                    names_J.append(["act"])

        J = ca.vertcat(J, pas_tau**2)
        names_J.append(["pas_tau"] * model.nbQ())
        if torque_tracking_as_objective:
            J = ca.vertcat(J, (tau - (mus_tau + pas_tau / scaling_factor[1])) ** 2)
            names_J.append(["tau"] * model.nbQ())
        else:
            if constr is None:
                constr = mus_tau + pas_tau
            else:
                constr = ca.vertcat(constr, (mus_tau + pas_tau))

    names_J = sum(names_J, [])
    ocp.cost.cost_type_0 = "NONLINEAR_LS"
    w = []
    for idx in range(J.numel()):
        w.append(weights[names_J[idx]])
    ocp.cost.W_0 = linalg.block_diag(np.diag(w))
    ocp.cost.yref_0 = np.zeros((ocp.cost.W_0.shape[0],))
    ocp.dims.ny_0 = ocp.cost.W_0.shape[0]
    ocp.model.cost_y_expr_0 = J.reshape((-1, 1))
    if not torque_tracking_as_objective:
        ocp.constraints.constr_type = "BGH"
        ocp.model.con_h_expr = constr
    return ocp, names_J


def _init_casadi_function(model):
    q_sym = ca.MX.sym("q", model.nbQ())
    qdot_sym = ca.MX.sym("qdot", model.nbQ())
    muscle_activations = ca.MX.sym("muscle_activation", model.nbMuscles())

    def muscle_joint_torque(activations_fct, q_fct, qdot_fct) -> ca.MX:
        muscles_states = model.stateSet()
        for k in range(model.nbMuscles()):
            muscles_states[k].setActivation(activations_fct[k])
        return model.muscularJointTorque(muscles_states, q_fct, qdot_fct).to_mx()

    mjt = muscle_joint_torque(muscle_activations, q_sym, qdot_sym)
    mjt_func = ca.Function("mjt_func", [muscle_activations, q_sym, qdot_sym], [mjt]).expand()
    return mjt_func


def _update_solver(ocp_solver, target, x0, q, qdot, tau=None, torque_as_objective=True, emg=None):
    # update initial guess
    # if x0 is not None:
    #     ocp_solver.set(0, "x", x0)
    # update targets
    # ocp_solver.cost_set(0, "yref", target)
    if not torque_as_objective:
        ocp_solver.constraints_set(0, "lh", tau[:, 0])
        ocp_solver.constraints_set(0, "uh", tau[:, 0])
        if emg is not None:
            ocp_solver.set(0, "p", np.vstack((q, qdot, emg)))
        else:
            ocp_solver.set(0, "p", np.vstack((q, qdot)))

    else:
        if emg is not None:
            ocp_solver.set(0, "p", np.vstack((q, qdot, tau, emg)))
        else:
            ocp_solver.set(0, "p", np.vstack((q, qdot, tau)))
    return ocp_solver


def _create_new_model(model, segments_names):
    model_empty = biorbd.Model()
    ordered_dof_prox_to_dist = []
    ordered_dof_prox_to_dist_idx = []
    q_non_zero_idx = []
    nb_dof = 0
    for i in range(model.nbSegment()):
        last_parent = None
        # if i == 0:
        #     ordered_dof_prox_to_dist = [model.segment(i).name().to_string()]
        if model.segment(i).name().to_string() in segments_names:
            rot = model.segment(i).seqR().to_string()
            trans = model.segment(i).seqT().to_string()
            if len(rot + trans) != 0:
                for j in range(len(rot + trans)):
                    q_non_zero_idx.append(nb_dof + j)
                model_empty = __add_segment_to_model(
                    model_empty, model.segment(i), rot, trans, name=model.segment(i).name().to_string() + "_init_dof"
                )
                last_parent = model.segment(i).name().to_string() + "_init_dof"
            nb_dof += len(rot + trans)
            rot = trans = "xyz"
            ordered_dof_prox_to_dist.append(model.segment(i).name().to_string())
            ordered_dof_prox_to_dist_idx.append([nb_dof + k for k in range(6)])
            nb_dof += 6
            model_empty = __add_segment_to_model(
                model_empty, model.segment(i), rot, trans, parent_str=last_parent, RT=biorbd.RotoTrans()
            )
        else:
            rot = model.segment(i).seqR().to_string()
            trans = model.segment(i).seqT().to_string()
            for j in range(len(rot + trans)):
                q_non_zero_idx.append(nb_dof + j)
            nb_dof += len(rot + trans)
            model_empty = __add_segment_to_model(model_empty, model.segment(i), rot, trans)
    for i, group in enumerate(model.muscleGroups()):
        name, insertion, origin = group.name().to_string(), group.insertion().to_string(), group.origin().to_string()
        model_empty.addMuscleGroup(name, origin, insertion)
        for m in range(group.nbMuscles()):
            model_empty.muscleGroups()[-1].addMuscle(model.muscleGroup(i).muscle(m))
    # for i, group in enumerate(model.ligaments()):
    #     name, insertion, origin = group.name().to_string(), group.insertion().to_string(), group.origin().to_string()
    #     model_empty.addMuscleGroup(name, origin, insertion)
    #     for m in range(group.nbMuscles()):
    #         model_empty.muscleGroups()[-1].addMuscle(model.muscleGroup(i).muscle(m))
    # if write_bioMod:
    #     biorbd.Writer.writeModel(model_empty, "model_simplified.bioMod")
    return model_empty, q_non_zero_idx, ordered_dof_prox_to_dist, ordered_dof_prox_to_dist_idx


def _compute_inverse_dynamics(model, q, qdot, qddot, segment_names, segment_idx, external_loads=None):
    external_biorbd_loads = None
    if external_loads:
        external_biorbd_loads = external_loads.to_biorbd_loads(model)
    Tau = model.InverseDynamics(q, qdot, qddot, external_biorbd_loads).to_array()
    translational_in_local = []
    rotationnal_in_local = []
    for j in range(len(segment_names)):
        translational_in_local.append(-Tau[segment_idx[j][:3]])
        rotationnal_in_local.append(-Tau[segment_idx[j][3:]])
    return translational_in_local, rotationnal_in_local


def _express_in_new_coordinate(trans, rot, new_application, all_global_jcs_old, inv_all_global_jcs_new):
    def get_homogeneous_vector(vector):
        if isinstance(vector, list):
            vector = np.array(vector)
        return np.append(vector, np.ones(1))

    global_trans = np.dot(all_global_jcs_old, get_homogeneous_vector(trans))
    global_rot = np.dot(all_global_jcs_old, get_homogeneous_vector(rot))
    application_point = [0, 0, 0, 1]
    application_point_global = all_global_jcs_old @ application_point
    trans_new_coordinate = np.dot(inv_all_global_jcs_new, global_trans)[:3]
    rot_new_coordinate = np.dot(inv_all_global_jcs_new, global_rot)[:3]
    application_point_local = (inv_all_global_jcs_new @ application_point_global)[:3]
    final_rotation = rot_new_coordinate + np.cross((application_point_local - new_application), trans_new_coordinate)
    return trans_new_coordinate, final_rotation


def __add_segment_to_model(model, segment, rot, trans, name=None, RT=None, parent_str=None):
    (name_tmp, parent_str_tmp, rot, trans, QRanges, QDotRanges, QDDotRanges, characteristics, RT_tmp) = (
        segment.name().to_string(),
        segment.parent().to_string(),
        rot,
        trans,
        [biorbd.Range(-3, 3)] * (len(rot) + len(trans)),
        [biorbd.Range(-3 * 10, 3 * 10)] * (len(rot) + len(trans)),
        [biorbd.Range(-3 * 100, 3 * 100)] * (len(rot) + len(trans)),
        segment.characteristics(),
        segment.localJCS(),
    )
    name = name if name else name_tmp
    RT = RT if RT else RT_tmp
    parent_str = parent_str if parent_str else parent_str_tmp
    model.AddSegment(name, parent_str, trans, rot, QRanges, QDotRanges, QDDotRanges, characteristics, RT)
    return model


def _compute_forces(model, q, qdot, act, segment_names, segment_idx, compound="muscle"):
    moment_arm = None
    if compound == "muscle":
        muscles_states = model.stateSet()
        for m in range(model.nbMuscles()):
            muscles_states[m].setActivation(act[m])
        moment_arm = model.muscularJointTorque(muscles_states, q, qdot).to_array()
    if compound == "ligament":
        moment_arm = model.ligamentJointTorque(q, qdot).to_array()
    idx_segment = []
    for k in range(model.nbSegment()):
        if model.segment(k).name().to_string() in segment_names:
            idx_segment.append(k)
    translational_in_local = []
    rotationnal_in_local = []
    for i in range(len(idx_segment)):
        translational_in_local.append(moment_arm[segment_idx[i][:3]])
        # translational_in_global.append(np.dot(model.globalJCS(idx_segment[i]).to_array(),
        #                                       np.array(local_forces.tolist() + [1])))
        rotationnal_in_local.append(moment_arm[segment_idx[i][3:]])
    return translational_in_local, rotationnal_in_local


class ExternalLoad:
    def __init__(self, point_of_application, applied_on_body, force, express_in_coordinate: str = "ground", name=""):
        self.point_of_application = point_of_application
        self.applied_on_body = applied_on_body
        self.express_in_coordinate = express_in_coordinate
        self.name = name
        self.force = force
        if isinstance(point_of_application, list):
            self.point_of_application = np.array(point_of_application)
        if self.point_of_application.shape[0] != 3:
            raise ValueError("The point of application must be a 3xn vector")
        if len(self.point_of_application.shape) != 1:
            raise ValueError("The point of application must be a one dimensional array")
        if isinstance(force, list):
            self.force = np.array(force)
        if self.force.shape[0] != 6:
            raise ValueError("The force must be a 6xn vector")
        if applied_on_body != express_in_coordinate and express_in_coordinate != "ground":
            raise NotImplementedError(
                "You can only either express the force in the ground or"
                " on the body where the force is applied."
                f"You have {applied_on_body} and {express_in_coordinate}"
            )


class ExternalLoads:
    def __init__(self):
        self.external_loads = []

    def add_external_load(self, point_of_application, applied_on_body, express_in_coordinate, load, name=None):
        self.external_loads.append(
            ExternalLoad(point_of_application, applied_on_body, load, express_in_coordinate, name)
        )

    def update_external_load_value(self, value: np.ndarray, name: str = None, idx: int = None):
        external_load = self.get_external_load(idx, name)
        if isinstance(value, list):
            value = np.array(value)
        if value.shape[0] != 6:
            raise ValueError(f"The load must be a 6xn vector." f" You provided a vector of shape {value.shape}.")
        external_load.force = value

    def get_external_load(self, idx: int = None, name: str = None):
        if (not idx and not name) or (idx and name):
            raise RuntimeError("Please provide either the force index or name.")
        if idx:
            return self.external_loads[idx]
        if name:
            return [ext_load for ext_load in self.external_loads if ext_load.name == name][0]

    def to_biorbd_loads(self, model):
        idx = list(range(len(self.external_loads)))
        return self.to_biorbd_load(idx=idx, model=model)

    def to_biorbd_load(self, model, name: (str, list) = None, idx: (int, list) = None):
        if (not idx and not name) or (idx and name):
            raise RuntimeError("Please provide either the force index or name.")
        ext_load = model.externalForceSet()
        for i, load in enumerate(self.external_loads):
            if i in idx or load.name == name:
                if load.express_in_coordinate == "ground":
                    ext_load.add(load.applied_on_body, load.force[:, 0], load.point_of_application)
                else:
                    ext_load.addInSegmentReferenceFrame(load.applied_on_body, load.force, load.point_of_application)
        return ext_load
