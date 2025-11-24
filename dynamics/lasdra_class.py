import numpy as np
from scipy.linalg import block_diag, sqrtm, solve_sylvester
from project_math.Adinv import Adinv
from project_math.adj import adj
from project_math.Tinv import Tinv
from .forward_kinematics_class import ForwardKinematics

# ─────────────────────────────────────────────────────────
# 수치 안정화 유틸
# ─────────────────────────────────────────────────────────
def _finite_clip(x, max_abs=1e6):
    """NaN/Inf 제거 + 절대값 클립."""
    x = np.asarray(x)
    x = np.nan_to_num(x, nan=0.0, posinf=max_abs, neginf=-max_abs)
    if max_abs is not None:
        x = np.clip(x, -max_abs, max_abs)
    return x


class LASDRA:
    def __init__(self, params):
        self.ODAR = params['ODAR']
        self.dof = params['LASDRA']['dof']
        self.numlinks = len(self.ODAR)

        # 상태
        self.q = np.zeros((self.dof, 1))
        self.dq = np.zeros((self.dof, 1))
        self.dV0 = np.array([[0], [0], [0], [0], [0], [9.81]])

        # 누적 변환 L, 외력 저장
        self.L = np.eye(6 * self.dof)
        self.Fodar = np.zeros((6 * self.numlinks, 1))

        # 내부 안전 한계
        self.MAX_ABS_DQ  = 24.0   # rad/s
        self.MAX_ABS_DDQ = 5.0    # rad/s^2

        # 초기화
        self.init_rotor_mapping(params)
        self.init_body_joint_screw_axes(params['LASDRA']['body_joint_screw_axes'])
        self.init_body_inertia_tensor(params['LASDRA']['inertia_matrix'])
        self.init_link_com_transforms(params)

        self.fk = ForwardKinematics(params)
        self.update_dynamics()

    # ───────────────────── 초기화 루틴들 ─────────────────────
    def init_rotor_mapping(self, params):
        eye_perm = np.block([[np.zeros((3, 3)), np.eye(3)],
                             [np.eye(3), np.zeros((3, 3))]])
        self.B_cell = []
        for odar in params['ODAR']:
            self.B_cell.append(eye_perm @ odar.B)
        self.B_blkdiag = block_diag(*self.B_cell)

    def init_body_joint_screw_axes(self, A_cell):
        self.S = self._to_6_by_dof(A_cell)
        self.A = np.zeros((6 * self.dof, self.dof))
        for j in range(self.dof):
            self.A[6*j:6*(j+1), j] = self.S[:, j]

    @staticmethod
    def _to_6_by_dof(A_cell):
        if isinstance(A_cell, np.ndarray):
            if A_cell.ndim == 2:
                if A_cell.shape[0] == 6:
                    return A_cell
                if A_cell.shape[1] == 6:
                    return A_cell.T
            if A_cell.ndim == 1 and (A_cell.size % 6 == 0):
                dof = A_cell.size // 6
                return A_cell.reshape(dof, 6).T
            raise ValueError(f"[LASDRA] Unsupported ndarray shape for screw axes: {A_cell.shape}")

        if isinstance(A_cell, (list, tuple)):
            cols = []
            for a in A_cell:
                arr = np.asarray(a).reshape(-1)
                if arr.size != 6:
                    raise ValueError(f"[LASDRA] Each screw axis must be 6x1; got shape {arr.shape}")
                cols.append(arr)
            return np.stack(cols, axis=1)
        raise TypeError(f"[LASDRA] body_joint_screw_axes must be ndarray or list; got {type(A_cell)}")

    def init_body_inertia_tensor(self, G_cell):
        self.G = block_diag(*G_cell)

    def init_link_com_transforms(self, params):
        transforms_per_joint = []
        joint_counts = []

        for odar in params['ODAR']:
            nj = len(odar.body_joint_screw_axes)
            joint_counts.append(nj)
            # 마지막 조인트 이전은 항등 (해당 링크의 마지막 조인트에만 CoM 적용)
            for _ in range(nj - 1):
                transforms_per_joint.append(Adinv(np.eye(4)))
            Tic = np.eye(4)
            Tic[:3, 3] = odar.joint_to_com
            transforms_per_joint.append(Adinv(Tic))

        AdAll = block_diag(*transforms_per_joint)

        d = self.dof
        Lnk = self.numlinks
        S_link = np.zeros((6 * Lnk, 6 * d))
        acc = 0
        for i, nj in enumerate(joint_counts):
            last_j = acc + nj - 1
            S_link[6*i:6*(i+1), 6*last_j:6*(last_j+1)] = np.eye(6)
            acc += nj

        # ← 여기서 경고가 떴었음: 지역적으로 억제
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            self.AdLinkCoMTransforms = _finite_clip(S_link @ AdAll)

        # EF 보정 (필요시)
        Tne = np.eye(4)
        Tne[:3, 3] = params['ODAR'][-1].length * np.array([1, 0, 0])
        Tn = np.eye(4)
        Tn[:3, 3] = params['ODAR'][-1].joint_to_com
        self.Tnce = Tinv(Tn) @ Tne

    # ───────────────────── 상태/동역학 업데이트 ─────────────────────
    def set_joint_states(self, q, dq):
        self.q = np.asarray(q)
        self.dq = _finite_clip(np.asarray(dq), max_abs=self.MAX_ABS_DQ)
        self.update_dynamics()

    def update_dynamics(self):
        Tijset = self.fk.compute_inter_joint_transition(self.q)

        # 누적 Adinv 블록 L
        self.L = np.eye(6 * self.dof)
        for i in range(1, self.dof):
            T = np.eye(4)
            for j in range(i - 1, -1, -1):
                T = Tijset[j + 1] @ T
                self.L[6 * i:6 * (i + 1), 6 * j:6 * (j + 1)] = Adinv(T)

        # ← 여기서도 경고가 뜨던 지점들: 지역 억제
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            V = _finite_clip(self.L @ self.A @ self.dq)

        self.ad_V = block_diag(*[adj(V[6*i:6*(i+1)]) for i in range(self.dof)])

        dV_base = np.zeros((6 * self.dof, 1))
        dV_base[:6] = Adinv(Tijset[0]) @ self.dV0

        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            self.Mass = _finite_clip(self.A.T @ self.L.T @ self.G @ self.L @ self.A)

        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            tmp = _finite_clip(self.G @ self.L @ self.ad_V - self.ad_V.T @ self.G @ self.L)
            self.Cori = _finite_clip(self.A.T @ self.L.T @ tmp @ self.A)

        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            self.Grav = _finite_clip(self.A.T @ self.L.T @ self.G @ self.L @ dV_base)

        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            self.D = _finite_clip(self.AdLinkCoMTransforms @ self.L @ self.A)

    # ───────────────────── 외력/토크 관련 ─────────────────────
    def get_joint_torque_from_odars(self):
        return self.D.T @ self.Fodar

    def set_odar_body_wrenches(self, F):
        self.Fodar = F

    def set_odar_body_wrench_from_thrust(self, thrust, iodar):
        Fi = self.B_cell[iodar] @ thrust
        self.Fodar[6*iodar:6*(iodar+1)] = Fi

    # ───────────────────── 적분기 ─────────────────────
    def get_next_joint_states(self, dt, tau_odar, external_wrench=None):
        self.step_count = getattr(self, "step_count", -1) + 1

        if external_wrench is None:
            tau_net = tau_odar
        else:
            R_EF = self.fk.compute_end_effector_frame(self.q)[:3, :3]
            blkR = block_diag(R_EF, R_EF)
            external_wrench_b = blkR.T @ np.vstack((external_wrench['torque'], external_wrench['force']))
            tau_net = tau_odar + self.fk.compute_end_effector_body_jacobian(self.q).T @ external_wrench_b

        # ddq = (M + mu I)^{-1} (tau - C dq - g)  with small regularization
        rhs = tau_net - self.Cori @ self.dq - self.Grav
        n = self.Mass.shape[0]
        mu = 1e-6 * (np.trace(self.Mass) / max(n, 1))

        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            ddq = np.linalg.solve(self.Mass + mu * np.eye(n), rhs)

        # ① 가속도 제한
        ddq = _finite_clip(ddq, max_abs=self.MAX_ABS_DDQ)
        # ② 반-암묵 오일러
        dq_next = self.dq + dt * ddq
        # ③ 속도 제한
        dq_next = _finite_clip(dq_next, max_abs=self.MAX_ABS_DQ)

        q_next = self.q + dt * dq_next
        return {'q': q_next, 'dq': dq_next}

    # (옵션) 느린 중점 적분기
    def passive_midpoint_integration(self, tau_net, ddq, dt):
        sMk = sqrtm(self.Mass)
        dsMk = solve_sylvester(sMk, sMk, self.Cori + self.Cori.T)
        A_bar = self.Mass / dt + 0.5 * (self.Cori - sMk @ dsMk)
        B_bar = tau_net - self.Grav + 2 * self.Mass @ self.dq / dt
        inv_A_B = np.linalg.solve(A_bar, B_bar)

        q_next = self.q + 0.5 * dt * inv_A_B
        sMkp = sqrtm(self.get_mass_matrix(q_next))
        dq_next = np.linalg.solve(sMkp, sMk @ (inv_A_B - self.dq))
        return {'q': q_next, 'dq': dq_next}

    def get_mass_matrix(self, q):
        Tijset = self.fk.compute_inter_joint_transition(q)
        L = np.eye(6 * self.dof)
        for i in range(1, self.dof):
            T = np.eye(4)
            for j in range(i - 1, -1, -1):
                T = Tijset[j + 1] @ T
                L[6 * i:6 * (i + 1), 6 * j:6 * (j + 1)] = Adinv(T)
        # 여기서도 잠재적 큰 수 → 지역 억제 & 클립
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            M = self.A.T @ L.T @ self.G @ L @ self.A
        return _finite_clip(M)
