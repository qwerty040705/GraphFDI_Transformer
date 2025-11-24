import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import numpy as np

# ────────────────────── math helpers ──────────────────────
def clip_inf_norm(vec: np.ndarray, max_abs: Optional[float] = None) -> np.ndarray:
    if max_abs is None:
        return vec
    s = np.linalg.norm(vec, ord=np.inf)
    return vec * (max_abs / (s + 1e-12)) if s > max_abs else vec

def vee(mat: np.ndarray) -> np.ndarray:
    return np.array([
        mat[2,1] - mat[1,2],
        mat[0,2] - mat[2,0],
        mat[1,0] - mat[0,1]
    ], dtype=float)

def orientation_error(R_now: np.ndarray, R_des: np.ndarray) -> np.ndarray:
    # e_R = 0.5 * (R_des^T R_now - R_now^T R_des)^vee
    return 0.5 * vee(R_des.T @ R_now - R_now.T @ R_des)

def clik_step(
    fk, q: np.ndarray, T_des: np.ndarray, dt: float,
    k_pos: float = 3.0, k_rot: float = 3.0, damp: float = 1e-5,
    dq_clip: float = 1.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Closed-loop IK(kinematic): v=[ω;v]=[k_rot*e_R ; k_pos*e_x], dq = J^+ v
    (※ 참조 q_des 생성용)
    """
    q = np.asarray(q).reshape(-1)
    T_now = fk.compute_end_effector_frame(q)
    Rn, pn = T_now[:3,:3], T_now[:3,3]
    Rd, pd = T_des[:3,:3], T_des[:3,3]

    e_r = orientation_error(Rn, Rd).reshape(3,1)
    e_x = (pd - pn).reshape(3,1)
    v6  = np.vstack([k_rot*e_r, k_pos*e_x])         # desired body twist (P only)

    J = np.asarray(fk.compute_end_effector_analytic_jacobian(q), dtype=float)  # 6xd
    JJt = J @ J.T
    pinv = J.T @ np.linalg.inv(JJt + damp*np.eye(6))  # DLS
    dq = pinv @ v6
    dq = clip_inf_norm(dq, dq_clip)
    q_next = q.reshape(-1,1) + dt*dq
    return q_next, v6, e_r, e_x, J

# ────────────────────── joint-PID controller ──────────────────────
@dataclass
class JointPIDGains:
    Kp: np.ndarray               # (dof, dof)
    Kd: np.ndarray               # (dof, dof)
    Ki: np.ndarray               # (dof, dof)
    i_err_limit: float = 0.3     # |integral error| clamp [rad·s]
    tau_clip: float = 5e4        # safety clip on torque

@dataclass
class ControllerOptions:
    use_actuator_delay: bool = False
    lpf_beta: float = 1.0        # 1.0 => no LPF; (0<β<=1)
    thrust_bound_scale: float = 50.0
    bypass_Blambda: bool = False   # ★ 추가: τ=Bλ 분배 건너뛰기 여부

class ControllerJointPIDWithCLIK:
    """
    1) CLIK으로 q_des, dq_des 생성
    2) 조인트 PID로 τ_des = Kp(qd-q) + Kd(dqd-dq) + Ki∫(qd-q)
    3) τ=B·λ 분배 → 각 링크 바디 렌치 적용 → 동역학 적분
    """
    def __init__(
        self,
        fk,                           # ForwardKinematics
        robot,                        # LASDRA
        external_controller,          # ExternalActuation (τ->λ)
        dof: int,
        link_count: int,
        dt: float,
        gains: JointPIDGains,
        opts: Optional[ControllerOptions] = None
    ):
        self.fk = fk
        self.robot = robot
        self.ext = external_controller
        self.dof = int(dof)
        self.link_count = int(link_count)
        self.dt = float(dt)
        self.g = gains
        self.opts = opts if opts is not None else ControllerOptions()

        # QP bounds 널널하게
        if hasattr(self.ext, "qp"):
            if "ub" in self.ext.qp and "lb" in self.ext.qp:
                self.ext.qp["ub"] *= self.opts.thrust_bound_scale
                self.ext.qp["lb"] *= self.opts.thrust_bound_scale

        # integral state
        self.i_err = np.zeros((self.dof, 1))

        # actuator internal state
        self.lam_state = np.zeros((8*self.link_count, 1))

    def _apply_integrator(self, e_q: np.ndarray, saturated: bool = False):
        # 간단한 anti-windup: 포화 추정 시 적분 동결
        if not saturated:
            self.i_err += e_q * self.dt
            # clamp each element
            self.i_err = np.clip(self.i_err, -self.g.i_err_limit, self.g.i_err_limit)

    def _distribute(self, tau_des: np.ndarray, label_vec_t: np.ndarray) -> np.ndarray:
        lam_cmd = self.ext.distribute_torque_lp(tau_des.reshape(-1)).reshape(-1,1)

        # fault mask
        faulty_mask = (np.asarray(label_vec_t).reshape(-1,1) == 0)
        lam_cmd[faulty_mask] = 0.0

        if self.opts.use_actuator_delay:
            lam_applied = self.lam_state.copy()
            self.lam_state = self.lam_state + self.opts.lpf_beta*(lam_cmd - self.lam_state)
        else:
            # no delay; just optional LPF
            self.lam_state = self.lam_state + self.opts.lpf_beta*(lam_cmd - self.lam_state)
            lam_applied = self.lam_state
        return lam_applied

    def step(
        self,
        T_des: np.ndarray,
        q_des_prev: np.ndarray,    # (dof,1)
        q_act_prev: np.ndarray,    # (dof,1)
        dq_act_prev: np.ndarray,   # (dof,1)
        label_vec_t: np.ndarray
    ) -> Dict:
        """
        returns:
          q_des_next, dq_des, q_next, dq_next, tau_des, lam_applied
        """
        # 1) CLIK → (q_des_next, dq_des)
        q_des_next, _, _, _, _ = clik_step(
            self.fk, q_des_prev, T_des, self.dt,
            k_pos=3.0, k_rot=3.0, damp=1e-5, dq_clip=1.2
        )
        dq_des = (q_des_next - q_des_prev)/self.dt

        # 2) joint PID (actual → desired)
        e_q  = q_des_next - q_act_prev
        ed_q = dq_des - dq_act_prev
        # tentative integral update (freeze if we detect saturation later)
        self._apply_integrator(e_q, saturated=False)

        tau_des = (self.g.Kp @ e_q) + (self.g.Kd @ ed_q) + (self.g.Ki @ self.i_err)
        tau_des = clip_inf_norm(tau_des, self.g.tau_clip)

        # 3) τ=B·λ 분배 (+fault masking / LPF)
        if self.opts.bypass_Blambda:
            # ★ 분배 건너뛰기: tau_des 그대로 동역학에 입력
            tau_apply = tau_des.reshape(-1)
            lam_applied = np.zeros_like(self.lam_state)
        else:
            lam_applied = self._distribute(tau_des, label_vec_t)
            # thrust→robot에 적용
            for iL in range(self.link_count):
                thrust_i = lam_applied[8*iL:8*(iL+1)]
                self.robot.set_odar_body_wrench_from_thrust(thrust_i, iL)
            tau_apply = self.robot.get_joint_torque_from_odars()

        # 4) 각 링크에 적용 → 동역학 적분
        nxt = self.robot.get_next_joint_states(self.dt, tau_apply)
        qn  = nxt['q']
        dqn = np.clip(nxt['dq'], -10.0, 10.0)
        if not (np.all(np.isfinite(qn)) and np.all(np.isfinite(dqn))):
            qn, dqn = np.asarray(q_act_prev), np.asarray(dq_act_prev)
        self.robot.set_joint_states(qn, dqn)

        return {
            "q_des_next": q_des_next, "dq_des": dq_des,
            "q_next": qn, "dq_next": dqn,
            "tau_des": tau_des, "lam_applied": lam_applied
        }

# ────────────────────── factory ──────────────────────
def make_default_joint_pid_gains(dof: int) -> JointPIDGains:
    # 강한 추종(고장 전 mm급) 기본 게인
    Kp = 500.0 * np.eye(dof, dtype=float)
    Kd =  60.0 * np.eye(dof, dtype=float)
    Ki =  10.0 * np.eye(dof, dtype=float)
    return JointPIDGains(Kp=Kp, Kd=Kd, Ki=Ki, i_err_limit=0.3, tau_clip=5e4)
