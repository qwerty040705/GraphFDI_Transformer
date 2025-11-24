# /Users/dnbn/code/transformer_fault_diagnosis/control/end_effector_impedance_controller.py

import numpy as np

def _limit_vec_norm(v, max_norm):
    n = float(np.linalg.norm(v))
    return v * (max_norm / n) if (n > max_norm and max_norm > 0) else v

def _unskew(R):
    return np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1]
    ], dtype=float).reshape(3, 1)

def _damped(prev, curr, alpha):
    return (1.0 - alpha) * prev + alpha * curr

def _col(x, n=None):
    x = np.asarray(x, dtype=float)
    if x.ndim == 0:
        x = x.reshape(1, 1)
    elif x.ndim == 1:
        x = x.reshape((-1, 1))
    elif x.ndim == 2:
        pass
    else:
        x = x.reshape((-1, 1))
    if n is not None and x.shape[0] != n:
        x = x.reshape((n, 1))
    return x

class EndEffectorImpedanceController:
    """
    End-Effector Impedance Controller with Nullspace PD Control
    (MATLAB 수학식과 동일하게 구현, Python에서는 수치 안정성 보강)
    """
    def __init__(self, params, lasdra_model):
        self.lasdra_model = lasdra_model
        self.fk = getattr(lasdra_model, "fk", None)
        if self.fk is None:
            raise RuntimeError("lasdra_model.fk (ForwardKinematics) required")

        efp = params['ef_impedance']

        # EE translational/rotational gains
        self.M_d_ef_inv    = np.linalg.inv(np.asarray(efp['desired_mass'], dtype=float))
        self.D_d_ef        = np.asarray(efp['desired_damping'], dtype=float)
        self.K_d_ef        = np.asarray(efp['desired_spring'], dtype=float)

        self.M_R_d_ef_inv  = np.linalg.inv(np.asarray(efp['desired_rotational_mass'], dtype=float))
        self.D_R_d_ef      = np.asarray(efp['desired_rotational_damping'], dtype=float)
        self.K_R_d_ef      = np.asarray(efp['desired_rotational_spring'], dtype=float)

        # nullspace PD
        self.K_nsp         = np.asarray(efp['K_nullspace'], dtype=float)
        self.B_nsp         = np.asarray(efp['B_nullspace'], dtype=float)

        # limits
        self.torque_limit_nominal = float(efp.get('torque_limit', 5.0e4))
        self.dof = int(params['dof'])
        self.torque_limit_ub = self.torque_limit_nominal * np.ones((self.dof,), dtype=float)
        self.torque_limit_lb = -self.torque_limit_ub

        # buffers
        self.T_ef_d_buffer_max = 4
        self.T_ef_buffer = {'position': [], 'orientation': []}
        self.e_prev = None
        self.de_prev = None
        self.num_diff_damp_coeff = 0.97

        # nullspace reference
        self.q_ref  = np.zeros((self.dof, 1), dtype=float)
        self.dq_ref = np.zeros((self.dof, 1), dtype=float)

        self.e_position_max = float(efp.get('e_position_max', 0.6))

        # prev J for dJ/dt
        self._J_prev = None

    # ---------------- API ----------------
    def update_desired_end_effector_pose(self, T_ef_d):
        x_ef_d = T_ef_d[:3, 3].astype(float)
        R_ef_d = T_ef_d[:3, :3].astype(float)
        if len(self.T_ef_buffer['position']) < self.T_ef_d_buffer_max:
            self.T_ef_buffer['position'].append(x_ef_d)
            self.T_ef_buffer['orientation'].append(R_ef_d)
        else:
            self.T_ef_buffer['position'] = self.T_ef_buffer['position'][1:] + [x_ef_d]
            self.T_ef_buffer['orientation'] = self.T_ef_buffer['orientation'][1:] + [R_ef_d]

    def set_nullspace_reference(self, q_ref, dq_ref=None):
        self.q_ref = _col(q_ref, self.dof)
        if dq_ref is None:
            self.dq_ref = np.zeros_like(self.q_ref)
        else:
            self.dq_ref = _col(dq_ref, self.dof)

    # ---------------- Core ----------------
    def get_control_output(self, dt, T_ef, joint_state):
        q  = _col(joint_state['q'],  self.dof)
        dq = _col(joint_state['dq'], self.dof)

        # dynamics
        M = np.asarray(self.lasdra_model.Mass, dtype=float)
        C = np.asarray(self.lasdra_model.Cori, dtype=float)
        G = _col(self.lasdra_model.Grav, self.dof)

        # jacobian
        J_e = np.asarray(self.fk.compute_end_effector_analytic_jacobian(q), dtype=float)
        if J_e.shape != (6, self.dof):
            raise RuntimeError(f"J_e shape must be (6,{self.dof}), got {J_e.shape}")

        # EE pose/err
        x_ef = _col(T_ef[:3, 3], 3)
        R_ef = T_ef[:3, :3].astype(float)
        R_d  = self.T_ef_buffer['orientation'][-1]
        x_d  = _col(self.T_ef_buffer['position'][-1], 3)

        # --- MATLAB 수식과 동일 ---
        e_r = R_ef @ (0.5 * _unskew(R_d.T @ R_ef - R_ef.T @ R_d))
        e_x = x_ef - x_d
        e   = np.vstack((e_r, e_x))

        # clip pos err
        e_clip = e.copy()
        e_clip[3:, :] = _limit_vec_norm(e_clip[3:, :], self.e_position_max)

        # de
        if self.e_prev is None:
            de = np.zeros_like(e)
        else:
            de_raw = (e - self.e_prev) / dt
            de = _damped(self.de_prev if self.de_prev is not None else np.zeros_like(e),
                         de_raw, self.num_diff_damp_coeff)
        self.e_prev = e
        self.de_prev = de

        # desired EE accel
        ddx_d = _col(self._get_desired_acc(dt), 6)

        # damped pseudo-inverse
        JJt = J_e @ J_e.T
        rho_square = 1e-6 + 0.02 * np.tanh(1.0 / (1e3 * np.linalg.det(JJt) + 1e-10))
        pinv_J = J_e.T @ np.linalg.inv(JJt + rho_square * np.eye(6))

        # task torque
        K_big = np.block([[self.K_R_d_ef, np.zeros((3,3))],
                          [np.zeros((3,3)), self.K_d_ef]])
        D_big = np.block([[self.D_R_d_ef, np.zeros((3,3))],
                          [np.zeros((3,3)), self.D_d_ef]])
        M_big_inv = np.block([[self.M_R_d_ef_inv, np.zeros((3,3))],
                              [np.zeros((3,3)), self.M_d_ef_inv]])

        dJdq = self._dJdq(J_e, dq, dt)
        rhs6 = M_big_inv @ (-D_big @ de - K_big @ e_clip) + ddx_d - dJdq
        tau_task = M @ (pinv_J @ rhs6) + C @ dq + G

        # nullspace torque (MATLAB과 동일: M 곱해줌)
        I = np.eye(self.dof)
        N = I - (pinv_J @ J_e)
        tau_null = self.K_nsp @ (self.q_ref - q) + self.B_nsp @ (self.dq_ref - dq)
        tau = tau_task + M @ (N @ tau_null)

        # saturation
        tau = np.clip(tau.reshape(-1), self.torque_limit_lb, self.torque_limit_ub).reshape(self.dof, 1)
        return tau

    # ---------------- helpers ----------------
    def _get_desired_acc(self, dt):
        xbuf = self.T_ef_buffer['position']
        if len(xbuf) <= 2:
            ddx_p = np.zeros((3,1))
        elif len(xbuf) == 3:
            coeff = np.array([1, -2, 1], dtype=float)
            ddx_p = (coeff[0]*_col(xbuf[-3],3) +
                     coeff[1]*_col(xbuf[-2],3) +
                     coeff[2]*_col(xbuf[-1],3)) / (dt**2)
        else:
            coeff = np.array([-1, 4, -5, 2], dtype=float)
            ddx_p = (coeff[0]*_col(xbuf[-4],3) +
                     coeff[1]*_col(xbuf[-3],3) +
                     coeff[2]*_col(xbuf[-2],3) +
                     coeff[3]*_col(xbuf[-1],3)) / (dt**2)
        return np.vstack((np.zeros((3,1)), ddx_p)).reshape(6, 1)

    def _dJdq(self, J, dq, dt):
        if self._J_prev is None:
            self._J_prev = J.copy()
            return np.zeros((6, 1))
        dJ = (J - self._J_prev) / dt
        self._J_prev = J.copy()
        return dJ @ dq
