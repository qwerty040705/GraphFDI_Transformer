import numpy as np

def _as_col(x, n=None, dtype=float):
    x = np.asarray(x, dtype=dtype)
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

class CentralizedController:
    """
    q→τ 조인트 PID 제어기.
    - param["LASDRA"]["multijoint_PID"]가 없어도 동작하도록 기본값을 자동 설정.
    - 기본값(없을 때):
        Kp = 150*I, Ki = 0*I, Kd = 10*I
        simplified 동일
        i_max = 50 (각 조인트), torque_limit = 5e4, max_acc_feedforward_ratio = 0.2
    """
    def __init__(self, param, lasdra_model):
        self.lasdra_model = lasdra_model

        las = param.get("LASDRA", {})
        dof = int(las.get("dof", 0))
        if dof <= 0:
            try:
                dof = int(np.asarray(getattr(lasdra_model, "q", np.zeros(1))).size)
            except Exception:
                raise RuntimeError("DoF 추정 실패: param['LASDRA']['dof']가 없고 모델에서도 유추 불가")

        pid = las.get("multijoint_PID", None)
        if pid is None:
            Kp_val, Ki_val, Kd_val = 150.0, 0.0, 10.0
            Kp = Kp_val * np.eye(dof)
            Ki = Ki_val * np.eye(dof)
            Kd = Kd_val * np.eye(dof)
            Ks = {"p": Kp.copy(), "i": Ki.copy(), "d": Kd.copy()}
            i_max = 50.0 * np.ones((dof, 1))
            torque_limit_nominal = float(las.get("torque_limit", 5.0e4))
            max_ff_ratio = 0.2
        else:
            def _get(obj, k, default=None):
                if isinstance(obj, dict):
                    return obj.get(k, default)
                return getattr(obj, k, default)

            def _to_mat(x, fallback_scalar):
                if x is None:
                    return float(fallback_scalar) * np.eye(dof)
                x = np.asarray(x, dtype=float)
                if x.ndim == 0:
                    return float(x) * np.eye(dof)
                if x.ndim == 1:
                    if x.size == 1:
                        return float(x.item()) * np.eye(dof)
                    return np.diag(x.reshape(-1))
                return x

            Kp = _to_mat(_get(pid, "p", None), 150.0)
            Ki = _to_mat(_get(pid, "i", None), 0.0)
            Kd = _to_mat(_get(pid, "d", None), 10.0)

            simp = _get(pid, "simplified", {})
            Kp_s = _to_mat(_get(simp, "p", None), 150.0)
            Ki_s = _to_mat(_get(simp, "i", None), 0.0)
            Kd_s = _to_mat(_get(simp, "d", None), 10.0)

            i_max = _get(pid, "i_max", 50.0)
            i_max = _as_col(i_max, dof)

            torque_limit_nominal = float(_get(pid, "torque_limit", las.get("torque_limit", 5.0e4)))
            max_ff_ratio = float(_get(pid, "max_acc_feedforward_ratio", 0.2))

            Ks = {"p": Kp_s, "i": Ki_s, "d": Kd_s}

        self.Kp = np.asarray(Kp, dtype=float)
        self.Ki = np.asarray(Ki, dtype=float)
        self.Kd = np.asarray(Kd, dtype=float)

        self.Kp_s = np.asarray(Ks["p"], dtype=float)
        self.Ki_s = np.asarray(Ks["i"], dtype=float)
        self.Kd_s = np.asarray(Ks["d"], dtype=float)

        self.dof = dof
        self.q_e_i = np.zeros((dof, 1))
        self.q_e_i_max = _as_col(i_max, dof)
        self.ddq_d = np.zeros((dof, 1))
        self.max_acc_feedforward_ratio = float(max_ff_ratio)

        self.torque_limit = {
            "nominal": float(torque_limit_nominal),
            "ub": float(torque_limit_nominal) * np.ones((dof, 1)),
            "lb": -float(torque_limit_nominal) * np.ones((dof, 1)),
        }

        self.q_prev = None
        self.dq_prev = None
        self.num_diff_damp_coeff = 0.97
        self.q_d = np.zeros((dof, 1))
        self.dq_d = np.zeros((dof, 1))

    # ---------------- desired setters ----------------
    def set_desired_joint_angle(self, q_d):
        self.q_d = _as_col(q_d, self.dof)

    def set_desired_joint_velocity(self, dq_d):
        self.dq_d = _as_col(dq_d, self.dof)

    def set_desired_state(self, joint_state):
        self.set_desired_joint_angle(joint_state["q"])
        self.set_desired_joint_velocity(joint_state["dq"])

    def set_desired_acceleration(self, ddq_d):
        self.ddq_d = _as_col(ddq_d, self.dof)

    # 편의 메서드: 현재 상태를 목표로 고정
    def set_desired_from_current(self, q, dq=None):
        self.set_desired_joint_angle(q)
        if dq is None:
            dq = np.zeros_like(self.q_d)
        self.set_desired_joint_velocity(dq)

    # ---------------- velocity helper ----------------
    def calculate_velocity(self, dt, q):
        q = _as_col(q, self.dof)
        if self.q_prev is None:
            dq = np.zeros_like(q)
        else:
            dq_raw = (q - self.q_prev) / float(dt)
            if self.dq_prev is None:
                dq = dq_raw
            else:
                dq = self.get_damped_value(self.dq_prev, dq_raw, self.num_diff_damp_coeff)
        self.q_prev = q
        self.dq_prev = dq
        return dq

    # ---------------- main control outputs ----------------
    def get_control_output(self, dt, q, dq=None):
        q = _as_col(q, self.dof)
        dq = self.calculate_velocity(dt, q) if dq is None else _as_col(dq, self.dof)

        q_e = q - self.q_d
        dq_e = dq - self.dq_d

        self.q_e_i += q_e * float(dt)
        self.q_e_i = self.bound(self.q_e_i, -self.q_e_i_max, self.q_e_i_max)

        M = np.asarray(self.lasdra_model.Mass, dtype=float)
        C = np.asarray(self.lasdra_model.Cori, dtype=float)
        G = _as_col(self.lasdra_model.Grav, self.dof)

        acc_ff = M @ self.ddq_d
        acc_ff_max = float(np.max(np.abs(acc_ff))) if acc_ff.size else 0.0
        if acc_ff_max > 0 and (acc_ff_max / self.torque_limit["nominal"] > self.max_acc_feedforward_ratio):
            acc_ff *= self.max_acc_feedforward_ratio / acc_ff_max

        torque = acc_ff + M @ (-self.Kp @ q_e - self.Ki @ self.q_e_i - self.Kd @ dq_e) + C @ dq + G
        torque = self.bound(torque, self.torque_limit["lb"], self.torque_limit["ub"])
        return torque

    def get_simplified_control_output(self, dt, q, dq=None):
        q = _as_col(q, self.dof)
        dq = self.calculate_velocity(dt, q) if dq is None else _as_col(dq, self.dof)

        q_e = q - self.q_d
        dq_e = dq - self.dq_d

        self.q_e_i += q_e * float(dt)
        self.q_e_i = self.bound(self.q_e_i, -self.q_e_i_max, self.q_e_i_max)

        G = _as_col(self.lasdra_model.Grav, self.dof)
        torque = -self.Kp_s @ q_e - self.Ki_s @ self.q_e_i - self.Kd_s @ dq_e + G
        torque = self.bound(torque, self.torque_limit["lb"], self.torque_limit["ub"])
        return torque

    def get_integral_control_output(self, dt, q, dq=None):
        q = _as_col(q, self.dof)
        dq = self.calculate_velocity(dt, q) if dq is None else _as_col(dq, self.dof)

        q_e = q - self.q_d
        self.q_e_i += q_e * float(dt)
        self.q_e_i = self.bound(self.q_e_i, -self.q_e_i_max, self.q_e_i_max)

        M = np.asarray(self.lasdra_model.Mass, dtype=float)
        torque = M @ (-self.Kp @ q_e - self.Ki @ self.q_e_i)
        torque = self.bound(torque, self.torque_limit["lb"], self.torque_limit["ub"])
        return torque

    # ---------------- compatibility wrappers ----------------
    # data_generate_new.py가 기대하는 이름들
    def compute_tau(self, dt, q, dq=None):
        """주 제어 출력 (full PID + 중력/코리올리 + feedforward)"""
        return self.get_control_output(dt, q, dq)

    def compute_tau_simplified(self, dt, q, dq=None):
        """간소화된 PID(+중력)"""
        return self.get_simplified_control_output(dt, q, dq)

    def compute_tau_integral(self, dt, q, dq=None):
        """P+I (논문 일부에서 사용하는 변형)"""
        return self.get_integral_control_output(dt, q, dq)

    # ---------------- helpers ----------------
    @staticmethod
    def get_damped_value(v_prev, v, damping):
        return (1 - float(damping)) * v_prev + float(damping) * v

    @staticmethod
    def bound(x, x_min, x_max):
        x = np.asarray(x, dtype=float)
        x_min = np.asarray(x_min, dtype=float)
        x_max = np.asarray(x_max, dtype=float)
        return np.maximum(x_min, np.minimum(x_max, x))

    def set_initial_state(self, joint_state):
        self.q_prev = _as_col(joint_state["q"], self.dof)
        self.dq_prev = _as_col(joint_state["dq"], self.dof)
