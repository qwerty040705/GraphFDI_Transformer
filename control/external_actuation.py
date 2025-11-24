# control/external_actuation.py

import numpy as np
import numpy.linalg as npl
from scipy.sparse import block_diag
from scipy.optimize import linprog
from dataclasses import dataclass
from control.selective_mapping import SelectiveMapping


class ExternalActuation:
    """
    외부 추력 분배기 (MATLAB 호환 + 논문식 per-link 옵션)

    주요 사용법:
      1) MATLAB식(기본): distribute_joint_linf(tau_joint)
         - 목적:  min ||λ||_inf
         - 제약:  D^T B λ = τ_joint,   lb ≤ λ ≤ ub
         - τ_joint: (d,) 또는 (d,1)  (d = dof)

      2) 논문식 per-link: distribute_wrench_linf(tau6L)
         - 각 링크 i에 대해:  min ||λ_i||_inf  s.t.  B_i^(τf) λ_i = τ_i,  lb_i ≤ λ_i ≤ ub_i
         - τ_i ∈ R^6 ([τ; f] 순서), τ6L는 (6L,) 스택

    공통:
      - B_i는 원래 [f; τ]라고 가정 → swap 행렬로 [τ; f] 정렬 후 사용
      - 선택적 selective mapping 후처리 지원(비활성화 권장: 등식 보장 깨질 수 있음)
    """
    MAT_CLIP = 1e8
    VEC_CLIP = 1e8

    @dataclass
    class PerLinkAlloc:
        Bp: np.ndarray       # (6×m_i)  [τ; f] 순서로 정렬된 B_i
        pinvBp: np.ndarray   # (m_i×6)
        N: np.ndarray        # (m_i×r)  nullspace 방향 (Bnsv를 B-nullspace로 정사영)
        lb: np.ndarray       # (m_i,)
        ub: np.ndarray       # (m_i,)

    def __init__(self, params_model, lasdra_model):
        # --- 모델/파라미터 ---
        self.lasdra = lasdra_model
        self.odars = params_model["ODAR"]
        self.nlinks = len(self.odars)

        # SelectiveMapping: 두 가지 시그니처 모두 대응
        try:
            self.smaps = [SelectiveMapping(self.odars[i]) for i in range(self.nlinks)]
        except TypeError:
            self.smaps = [SelectiveMapping(self.nlinks, i) for i in range(self.nlinks)]
        self.apply_selective_mapping = False  # 등식 보장을 원하면 False 유지 권장

        # [τ; f] ↔ [f; τ] swap
        self._swap = np.block([
            [np.zeros((3, 3)), np.eye(3)],
            [np.eye(3), np.zeros((3, 3))]
        ])

        # --- 전역 B (블록대각, [τ; f] 정렬) ---
        self.B_blkdiag = self._build_B_blkdiag(params_model)    # (6L × m)
        self.B_blkdiag = self._sanitize_mat(self.B_blkdiag, clip=self.MAT_CLIP)
        self.m_total = self.B_blkdiag.shape[1]

        # --- per-link 할당기(논문식용) ---
        self.allocs = self._build_per_link_allocators()

        # --- 경계/LP 변수 (MATLAB식 전역 LP용) ---
        self.ub_global = self._generate_thrust_bounds(params_model)  # (m,)
        self.lb_global = -self.ub_global
        self._init_joint_linf_lp()

        self.lambda_set_prev = None  # warm start 용도(옵션)

    # ───────────── 유틸 ─────────────
    @staticmethod
    def _sanitize_mat(M, clip=1e8):
        M = np.asarray(M, dtype=float)
        M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
        if clip is not None:
            M = np.clip(M, -clip, clip)
        return M

    @staticmethod
    def _sanitize_vec(v, clip=1e8):
        v = np.asarray(v, dtype=float).reshape(-1)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        if clip is not None:
            v = np.clip(v, -clip, clip)
        return v

    @classmethod
    def _safe_matvec(cls, M, v, mat_clip=None, vec_clip=None):
        M = cls._sanitize_mat(M, clip=mat_clip if mat_clip is not None else cls.MAT_CLIP)
        v = cls._sanitize_vec(v, clip=vec_clip if vec_clip is not None else cls.VEC_CLIP)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            y = M @ v
        return cls._sanitize_vec(y, clip=vec_clip if vec_clip is not None else cls.VEC_CLIP)

    # ───────────── 내부 구성 ─────────────
    def _build_B_blkdiag(self, params_model):
        # 각 링크 B_i: [f; τ] → [τ; f] 로 정렬
        B_blocks = [self._swap @ np.asarray(odar.B, dtype=float) for odar in params_model["ODAR"]]
        return block_diag(B_blocks).toarray()

    def _generate_thrust_bounds(self, params_model):
        ub = []
        for odar in params_model["ODAR"]:
            ub.extend([float(odar.max_thrust)] * np.asarray(odar.B).shape[1])
        return np.asarray(ub, dtype=float)

    def _init_joint_linf_lp(self):
        """
        전역 LP (MATLAB distributeTorqueLP 동일 구조)
          min s
          s.t. D^T B λ = τ_joint
               -s ≤ λ_i ≤ s
               lb ≤ λ ≤ ub
        """
        m = self.m_total
        self.lp = {}
        # 목적: [zeros(m), 1] @ [λ; s]
        self.lp["c"] = np.concatenate([np.zeros(m), [1.0]])
        # -s ≤ λ ≤ s  →  [ I | -1] [λ;s] ≤ 0  및 [-I | -1][λ;s] ≤ 0
        self.lp["A_box"] = np.vstack([
            np.hstack([ np.eye(m), -np.ones((m, 1))]),
            np.hstack([-np.eye(m), -np.ones((m, 1))]),
        ])
        self.lp["b_box"] = np.zeros(2 * m)
        # 경계: lb ≤ λ ≤ ub,  s ≥ 0
        self.lp["bounds"] = list(zip(np.append(self.lb_global, 0.0),
                                     np.append(self.ub_global, np.inf)))

    def _build_per_link_allocators(self):
        allocs = []
        for odar in self.odars:
            # [τ; f] 정렬된 per-link B
            Bp = self._swap @ np.asarray(odar.B, dtype=float)      # (6×m_i)
            Bp = self._sanitize_mat(Bp, clip=self.MAT_CLIP)
            pinvBp = npl.pinv(Bp, rcond=1e-9)

            # Bnsv 기반 nullspace 방향을 B-nullspace로 정사영
            u = np.asarray(odar.Bnsv['upper'], dtype=float).reshape(-1, 1)  # (m_i,1)
            v = np.asarray(odar.Bnsv['lower'], dtype=float).reshape(-1, 1)  # (m_i,1)
            I = np.eye(Bp.shape[1])
            Pn = I - pinvBp @ Bp
            N = Pn @ np.hstack([u, v])
            keep = [j for j in range(N.shape[1]) if npl.norm(N[:, j]) > 1e-8]
            N = N[:, keep] if keep else np.zeros((Bp.shape[1], 0))

            # 박스 제약
            ub = float(odar.max_thrust) * np.ones((Bp.shape[1],), dtype=float)
            lb = -ub
            allocs.append(self.PerLinkAlloc(Bp= Bp, pinvBp= pinvBp, N= N, lb= lb, ub= ub))
        return allocs

    # ───────────── per-link l∞ + box + nullspace (논문식) ─────────────
    @staticmethod
    def _per_link_linf_min(lambda0, N, lb, ub):
        """
        min s
        s.t.  -s ≤ λ0 + Nα ≤ s
              lb ≤ λ0 + Nα ≤ ub
        """
        m = lambda0.size
        r = N.shape[1] if N.size else 0
        if r == 0:
            return np.clip(lambda0, lb, ub)

        A_ub = np.vstack([
            np.hstack([  N, -np.ones((m, 1))]),   #  Nα - s ≥ -λ0
            np.hstack([ -N, -np.ones((m, 1))]),   # -Nα - s ≥  λ0
            np.hstack([  N,  np.zeros((m, 1))]),  #  Nα ≥ lb-λ0
            np.hstack([ -N,  np.zeros((m, 1))]),  # -Nα ≥ -(ub-λ0)
        ])
        b_ub = np.concatenate([
            -lambda0,
             lambda0,
            (lb - lambda0),
           -(ub - lambda0),
        ])
        c = np.zeros(r + 1); c[-1] = 1.0
        try:
            res = linprog(c=c, A_ub=A_ub, b_ub=b_ub,
                          bounds=[(None, None)] * r + [(0, None)],
                          method="highs")
            if res is not None and res.success:
                alpha = res.x[:r]
                lam = lambda0 + (N @ alpha)
                return np.clip(lam, lb, ub)
        except Exception:
            pass
        return np.clip(lambda0, lb, ub)

    # ───────────── 공개 API ① : MATLAB식 전역 LP (D 포함) ─────────────
    def distribute_joint_linf(self, tau_joint):
        """
        MATLAB distributeTorqueLP와 동일:
          min ||λ||_inf
          s.t.  D^T B λ = τ_joint,   lb ≤ λ ≤ ub

        tau_joint : (d,) or (d,1)  (joint torque vector)
        return    : (m,)  (all rotor thrusts)
        """
        tau_joint = self._sanitize_vec(tau_joint, clip=self.VEC_CLIP)
        D = self._sanitize_mat(self.lasdra.D, clip=self.MAT_CLIP)      # (6L×d)
        D_s  = self._sanitize_mat(D, clip=self.MAT_CLIP)
        B_s  = self._sanitize_mat(self.B_blkdiag, clip=self.MAT_CLIP)
        Aeq  = self._sanitize_mat(D_s.T @ B_s, clip=self.MAT_CLIP)  # (d×m)


        # 확장변수 [λ; s]
        A_eq_ext = np.hstack([Aeq, np.zeros((Aeq.shape[0], 1))])
        try:
            res = linprog(
                c=self.lp["c"],
                A_ub=self.lp["A_box"], b_ub=self.lp["b_box"],
                A_eq=A_eq_ext, b_eq=tau_joint,
                bounds=self.lp["bounds"],
                method="highs",
            )
        except Exception:
            res = None

        if (res is None) or (not res.success):
            # 간단한 fallback: 최소제곱 후 박스 클립(등식 정확도는 떨어질 수 있음)
            lam_ls = npl.pinv(Aeq, rcond=1e-9) @ tau_joint
            lam_ls = np.clip(lam_ls, self.lb_global, self.ub_global)
            lam = self._sanitize_vec(lam_ls, clip=self.VEC_CLIP)
        else:
            lam = self._sanitize_vec(res.x[:-1], clip=self.VEC_CLIP)

        if self.apply_selective_mapping:
            # 주의: 후처리는 등식을 깨뜨릴 수 있음. False 권장.
            Fodar = self._safe_matvec(self.B_blkdiag, lam)
            lam = self._apply_selective_mapping_from_F(Fodar)

        self.lambda_set_prev = lam.copy()
        return lam

    # ───────────── 공개 API ② : 논문식 per-link (D 미포함) ─────────────
    def distribute_wrench_linf(self, tau6L):
        """
        각 링크별 6D 렌치 τ_i=[τ; f]가 주어졌을 때,
        링크 i마다  min ||λ_i||_inf  s.t.  B_i^(τf) λ_i = τ_i,  lb_i ≤ λ_i ≤ ub_i
        """
        tau6L = self._sanitize_vec(tau6L, clip=self.VEC_CLIP)
        assert tau6L.size == 6 * self.nlinks, f"tau size {tau6L.size} but expected 6*{self.nlinks}"

        lam_list = []
        for i, alloc in enumerate(self.allocs):
            ti = tau6L[i * 6:(i + 1) * 6]                 # (6,)
            lambda0 = alloc.pinvBp @ ti                   # min-norm
            lam_i   = self._per_link_linf_min(lambda0, alloc.N, alloc.lb, alloc.ub)
            lam_list.append(lam_i)

        lam = np.concatenate(lam_list, axis=0)
        lam = self._sanitize_vec(lam, clip=self.VEC_CLIP)

        if self.apply_selective_mapping:
            lam = self._apply_selective_mapping_from_tau(tau6L)

        self.lambda_set_prev = lam.copy()
        return lam

    # ───────────── 선택적 후처리 (등식 보장 깨질 수 있음) ─────────────
    def _apply_selective_mapping_from_tau(self, tau6L):
        thrusts = []
        for i in range(self.nlinks):
            ti = tau6L[i * 6:(i + 1) * 6]
            wrench = {"torque": ti[:3], "force": ti[3:]}  # [τ; f]
            thrusts.append(self.smaps[i].get_adjusted_thrust(wrench))
        return self._sanitize_vec(np.concatenate(thrusts, axis=0), clip=self.VEC_CLIP)

    def _apply_selective_mapping_from_F(self, Fodar):
        thrusts = []
        for i in range(self.nlinks):
            Fi = Fodar[i * 6:(i + 1) * 6]
            if Fi.shape[0] != 6:
                Fi = np.zeros(6, dtype=float)
            wrench = {"torque": Fi[:3], "force": Fi[3:]}  # [τ; f]
            thrusts.append(self.smaps[i].get_adjusted_thrust(wrench))
        return self._sanitize_vec(np.concatenate(thrusts, axis=0), clip=self.VEC_CLIP)

    # ───────────── 디버그 헬퍼 ─────────────
    def reconstruct_joint_torque(self, lam):
        """ D^T B λ 계산(공동식 등식 잔차 확인용) """
        D = self._sanitize_mat(self.lasdra.D, clip=self.MAT_CLIP)
        lam = self._sanitize_vec(lam, clip=self.VEC_CLIP)
        return self._safe_matvec(D.T @ self.B_blkdiag, lam)

    def reconstruct_link_wrench(self, lam):
        """ B λ (링크 렌치 복원) """
        lam = self._sanitize_vec(lam, clip=self.VEC_CLIP)
        return self._safe_matvec(self.B_blkdiag, lam)
