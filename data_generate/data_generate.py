# data_generate/data_generate_new.py
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import multiprocessing as mp
import sys, io, time, contextlib, re, glob
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.sparse import block_diag as sp_block_diag

# ──────────────────── Project imports ────────────────────
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dynamics.forward_kinematics_class import ForwardKinematics
from dynamics.lasdra_class import LASDRA
try:
    from fault_injection_fast import inject_faults_fast as inject_faults
except Exception:
    from fault_injection import inject_faults
from parameters import get_parameters
from parameters_model import parameters_model
from control.external_actuation import ExternalActuation      # 추력 분배기
from control.centralized_controller import CentralizedController  # PID (q→τ)

# ---------------- Sharding config ----------------
SHARD_SIZE = int(os.environ.get("SHARD_SIZE", "1000"))  # ✨ 1000개마다 저장
USE_COMPRESSED = True

def _to_f32(x):
    x = np.asarray(x)
    if np.issubdtype(x.dtype, np.floating):
        return x.astype(np.float32, copy=False)
    return x

def _max_existing_shard_index(save_dir: str) -> int:
    """
    save_dir 안의 fault_dataset_shard_*.npz 중 최대 인덱스 반환. (없으면 0)
    """
    pat = os.path.join(save_dir, "fault_dataset_shard_*.npz")
    max_idx = 0
    for p in glob.glob(pat):
        m = re.search(r"fault_dataset_shard_(\d{5})\.npz$", os.path.basename(p))
        if m:
            idx = int(m.group(1))
            if idx > max_idx:
                max_idx = idx
    return max_idx

# ---------------- Utils ----------------
def _quiet_call(func, *args, **kwargs):
    with io.StringIO() as buf, contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        return func(*args, **kwargs)

def _format_hms(sec: float) -> str:
    sec = max(0, int(sec))
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

def _progress_bar(done, total, prefix="Generating", start_time=None):
    bar_len = 100
    frac = done / total if total else 1.0
    bar = "#" * int(bar_len * frac) + "-" * (bar_len - int(bar_len * frac))
    msg = f"\r{prefix} [{bar}] {frac*100:5.1f}% ({done}/{total})"
    if start_time:
        elapsed = time.time() - start_time
        remain  = (total - done) * elapsed / max(done, 1e-9)
        msg += f" | elapsed {_format_hms(elapsed)} | ETA {_format_hms(remain)}"
    print(msg, end="", flush=True)
    if done >= total:
        print()

def _is_finite(x): return np.all(np.isfinite(x))

def clip_inf_norm(vec, max_abs=None):
    if max_abs is None: return vec
    s = np.linalg.norm(vec, ord=np.inf)
    return vec * (max_abs / (s + 1e-12)) if s > max_abs else vec

# ---------------- Math helpers ----------------
def random_unit():
    v = np.random.randn(3); n = np.linalg.norm(v)
    return v / (n + 1e-12)

def random_so3():
    return R.from_rotvec(random_unit() * np.random.rand() * 2 * np.pi)

def align_z_to(vec_z):
    z = vec_z / (np.linalg.norm(vec_z) + 1e-12)
    x_tmp = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    y = np.cross(z, x_tmp); y /= np.linalg.norm(y) + 1e-12
    x = np.cross(y, z)
    Rm = np.stack([x, y, z], axis=1)
    return R.from_matrix(Rm)

def sample_on_cone(axis, alpha_max):
    u = random_unit()
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    cos_max = np.cos(alpha_max)
    while True:
        if np.dot(u, axis) >= cos_max:
            return u
        u = random_unit()

# ------------------------------------------------------------
# Desired SE(3) 궤적 생성기
# ------------------------------------------------------------
def generate_random_se3_series(
    fk_solver, q0, link_count, link_length, T=200,
    max_pos_step=0.004, max_rot_step=0.01, max_try=30,
):
    ℓ = float(link_length); n = link_count
    T0 = fk_solver.compute_end_effector_frame(q0.reshape(-1))
    p_cur = T0[:3, 3].copy()
    R_cur = R.from_matrix(T0[:3, :3])
    T_series = [T0]; r_max = n * ℓ

    for _ in range(1, T):
        for _try in range(max_try):
            p_cand = p_cur + np.random.uniform(-max_pos_step, max_pos_step, size=3)
            r = np.linalg.norm(p_cand)
            if n == 1:
                p_cand = p_cand / (r + 1e-12) * ℓ
            else:
                if r > r_max:
                    p_cand *= (r_max / (r + 1e-12))
                    r = r_max

            if n == 1:
                roll = np.random.uniform(-max_rot_step, max_rot_step)
                R_new = align_z_to(p_cand / ℓ) * R.from_rotvec(np.array([0, 0, roll]))
            elif n == 2:
                if r <= ℓ + 1e-12:
                    R_new = random_so3()
                elif r >= 2*ℓ - 1e-12:
                    roll = np.random.uniform(-max_rot_step, max_rot_step)
                    R_new = align_z_to(p_cand / r) * R.from_rotvec(np.array([0, 0, roll]))
                else:
                    alpha_max = np.arccos(np.clip(r / (2*ℓ), -1.0, 1.0))
                    z_dir = sample_on_cone(p_cand / r, alpha_max)
                    roll = np.random.uniform(-max_rot_step, max_rot_step)
                    R_new = align_z_to(z_dir) * R.from_rotvec(np.array([0, 0, roll]))
            else:
                s = n*ℓ - r
                if s >= 2*ℓ - 1e-12:
                    R_new = random_so3()
                elif s <= 1e-12:
                    roll = np.random.uniform(-max_rot_step, max_rot_step)
                    R_new = align_z_to(p_cand / r) * R.from_rotvec(np.array([0, 0, roll]))
                else:
                    alpha_max = np.arccos(
                        np.clip((r**2 + ℓ**2 - (n-1)**2 * ℓ**2) / (2*r*ℓ), -1.0, 1.0)
                    )
                    z_dir = sample_on_cone(p_cand / r, alpha_max)
                    roll = np.random.uniform(-max_rot_step, max_rot_step)
                    R_new = align_z_to(z_dir) * R.from_rotvec(np.array([0, 0, roll]))

            T_new = np.eye(4)
            T_new[:3, :3] = R_new.as_matrix()
            T_new[:3, 3] = p_cand
            if np.all(np.isfinite(T_new)):
                T_series.append(T_new)
                p_cur, R_cur = p_cand, R_new
                break
        else:
            T_series.append(T_series[-1].copy())
    return T_series

# ---------------- helpers: per-link transforms ----------------
def _exp_se3_from_Aset(fk_solver, j_idx, theta):
    xi = fk_solver.Aset[:, j_idx]
    return ForwardKinematics._exp_se3(xi, float(theta))

def compute_link_relatives_pure(fk_solver, q_vec):
    q_flat = np.asarray(q_vec).reshape(-1)
    link_T = []
    joint_counts = [len(odar.body_joint_screw_axes) for odar in fk_solver.ODAR]
    j0 = 0
    for k, nj in enumerate(joint_counts):
        T_link = np.eye(4)
        for j in range(nj):
            T_link = T_link @ _exp_se3_from_Aset(fk_solver, j0 + j, q_flat[j0 + j])
        T_link = T_link @ fk_solver.T_joint_to_joint_set[k]
        link_T.append(T_link)
        j0 += nj
    return link_T, joint_counts

def compute_link_cumulative_from_rel(link_rel_list):
    cum = []
    T = np.eye(4)
    for Trel in link_rel_list:
        T = T @ Trel
        cum.append(T.copy())
    return cum

# ---------------- CLIK & EE-PD → τ (optional 시각화/검증용) ----------------
def vee(mat):
    return np.array([
        mat[2,1] - mat[1,2],
        mat[0,2] - mat[2,0],
        mat[1,0] - mat[0,1]
    ], dtype=float)

def orientation_error(R_now, R_des):
    return 0.5 * vee(R_des.T @ R_now - R_now.T @ R_des)

def clik_step(fk, q, T_des, dt, k_pos=2.0, k_rot=2.0, damp=1e-3, dq_clip=1.5):
    """
    Closed-loop IK: v = [ω; v] = [k_rot*e_R ; k_pos*e_x], dq = J^+ v
    dq_clip: 관절속도 무한노름 상한 (rad/s). 초반 램프용으로 외부에서 주입.
    """
    q_flat = np.asarray(q).reshape(-1)          # (d,)
    T_now = fk.compute_end_effector_frame(q_flat)
    Rn, pn = T_now[:3,:3], T_now[:3,3]
    Rd, pd = T_des[:3,:3], T_des[:3,3]

    e_r = orientation_error(Rn, Rd).reshape(3,1)
    e_x = (pd - pn).reshape(3,1)
    v6  = np.vstack([k_rot*e_r, k_pos*e_x])     # desired body twist (6x1)

    J = np.asarray(fk.compute_end_effector_analytic_jacobian(q_flat), dtype=float)  # 6xd
    JJt = J @ J.T
    pinv = J.T @ np.linalg.inv(JJt + damp*np.eye(6))  # d×6

    dq = pinv @ v6                                   # (d,1)
    dq = clip_inf_norm(dq, dq_clip)                  # ★ 외부 주입 상한
    q_next_col = q_flat.reshape(-1,1) + dt*dq        # (d,1)
    return q_next_col.reshape(-1), v6, e_r, e_x, J   # 반환은 1D (d,)

def ee_pd_wrench(e_r, e_x, de_r, de_x,
                 Kp_r=np.diag([40,40,40]), Kd_r=np.diag([6,6,6]),
                 Kp_x=np.diag([900,900,900]), Kd_x=np.diag([90,90,90])):
    # F_des = [τ; f] (주의: D와 정렬) = [Kp_r e_r + Kd_r de_r ; Kp_x e_x + Kd_x de_x]
    tau = Kp_r @ e_r + Kd_r @ de_r
    f   = Kp_x @ e_x + Kd_x @ de_x
    return np.vstack([tau, f])

# ---------------- B_blkdiag helper (D와 정렬)
def build_B_blkdiag(params_model):
    """D가 [τ; f] 순서를 쓰므로, B(기본 [f; τ])에 swap(permutation) 적용"""
    eye_perm = np.block([
        [np.zeros((3, 3)), np.eye(3)],
        [np.eye(3), np.zeros((3, 3))]
    ])
    B_blocks = [eye_perm @ np.asarray(odar.B, dtype=float) for odar in params_model['ODAR']]
    return sp_block_diag(B_blocks).toarray()

# ---------------- Fault helpers ----------------
def first_fault_time_from_labels(label_matrix: np.ndarray) -> int:
    T = label_matrix.shape[0]
    for t in range(T):
        if np.any(label_matrix[t] == 0):
            return t
    return 10**9  # effectively "no fault"

def onset_vector_from_labels(label_matrix: np.ndarray) -> np.ndarray:
    T, M = label_matrix.shape
    onset = -np.ones((M,), dtype=np.int32)
    for j in range(M):
        idx = np.where(label_matrix[:, j] == 0)[0]
        if idx.size > 0:
            onset[j] = int(idx[0])
    return onset

# ---------------- One sample generator ----------------
def generate_one_sample(link_count, T=1000, epsilon_scale=0.0, dt=0.01, seed=None, bypass_Blambda=True):
    """
    생성 파이프라인:
      1) CLIK로 q_des(t) 궤적 생성
      2) CentralizedController 로 joint-space τ_joint(t) 산출 (q_des 추종)
      3) 외부 추력 분배기(전역 LP): D^T B λ = τ_joint  → λ(t)
      4) 실제 적용 토크: τ_apply = D^T B λ
      5) 고장(mask) 적용 후 동역학 적분

    ★ FREEZE MODE: t₀ 이후엔 LP를 다시 풀지 않는다. t₀ 직전 λ를 복사해
      고장 컬럼만 0으로 만든 뒤 그 λ를 마지막까지 그대로 적용.
    """
    if seed is not None:
        np.random.seed(seed)

    # ----- 모델 파라미터 구성 -----
    base_param = get_parameters(link_count)
    base_param['ODAR'] = base_param['ODAR'][:link_count]
    screw_axes, inertias = [], []
    for odar in base_param['ODAR']:
        screw_axes.extend(odar.body_joint_screw_axes)
        inertias.extend(odar.joint_inertia_tensor)
    base_param['LASDRA'].update(
        body_joint_screw_axes=screw_axes,
        inertia_matrix=inertias,
        dof=len(screw_axes)
    )
    model_param = _quiet_call(parameters_model, mode=0, params_prev=base_param)

    robot = LASDRA(model_param)
    fk_solver = robot.fk
    ext_act   = ExternalActuation(model_param, robot)                # ★ 전역 LP (D 포함)
    cc        = CentralizedController(model_param, robot)            # ★ 중앙집중 PID

    dof       = model_param['LASDRA']['dof']
    link_len  = float(model_param['ODAR'][0].length)

    # ----- 원하는 SE3 궤적 -----
    q0 = (np.random.rand(dof, 1) * 2 - 1) * 0.2 * np.pi
    T_des_series = generate_random_se3_series(
        fk_solver, q0, link_count, link_len, T=T,
        max_pos_step=0.0005, max_rot_step=0.001
    )

    # ----- B_blkdiag (D와 순서 일치) -----
    B_blk = build_B_blkdiag(model_param)                 # (6L)×(8L)

    # ----- 고장 시나리오 -----
    lam_dummy = np.zeros((T, B_blk.shape[1]))
    _, type_matrix, label_matrix, t0, idx, which_mask, onset_idx_raw = inject_faults(
        lam_dummy, epsilon_scale=epsilon_scale, return_labels=True
    )
    label_matrix = np.asarray(label_matrix, dtype=np.int32)  # (T, 8*link_count)
    which_mask = np.asarray(which_mask, dtype=np.int32)      # (8*link_count,)
    first_fault_t = first_fault_time_from_labels(label_matrix)
    onset_vec = onset_vector_from_labels(label_matrix)       # (8L,)

    # ==== FREEZE MODE 상태변수 ====
    lam_prev: np.ndarray | None = None     # t₀ 직전 분배
    lam_frozen: np.ndarray | None = None   # t₀~끝까지 고정 분배
    fault_has_triggered = False
    faulty_cols = np.where(which_mask == 1)[0]  # inject에서 고장 모터 컬럼(들)
    t0_fault = int(first_fault_t) if first_fault_t < 10**9 else None

    # ----- 버퍼 -----
    q_des  = np.zeros((T, dof, 1)); dq_des = np.zeros_like(q_des)
    q_act  = np.zeros_like(q_des);   dq_act = np.zeros_like(q_des)

    q_des[0] = q0; q_act[0] = q0
    robot.set_joint_states(q_act[0], np.zeros_like(q_act[0]))
    cc.set_initial_state({"q": q_act[0], "dq": np.zeros_like(q_act[0])})

    desired_ee = np.zeros((T, 4, 4))
    actual_ee  = np.zeros((T, 4, 4))
    desired_ee[0] = T_des_series[0]                 # ★ 레퍼런스 궤적 그대로 저장
    actual_ee[0]  = fk_solver.compute_end_effector_frame(q_act[0][:,0])

    desired_link_rel = np.zeros((T, link_count, 4, 4))
    actual_link_rel  = np.zeros((T, link_count, 4, 4))
    desired_link_cum = np.zeros((T, link_count, 4, 4))
    actual_link_cum  = np.zeros((T, link_count, 4, 4))

    # ★ 링크별 렌치 버퍼: w_link[t, ℓ, :] = 해당 시점 ℓ번째 링크의 [τ; f] (6차원)
    link_wrench = np.zeros((T, link_count, 6), dtype=np.float32)

    lr0_des, joint_counts = compute_link_relatives_pure(fk_solver, q_des[0][:,0])
    lr0_act, _            = compute_link_relatives_pure(fk_solver, q_act[0][:,0])
    lc0_des = compute_link_cumulative_from_rel(lr0_des)
    lc0_act = compute_link_cumulative_from_rel(lr0_act)
    for k in range(link_count):
        desired_link_rel[0, k] = lr0_des[k]; desired_link_cum[0, k] = lc0_des[k]
        actual_link_rel[0, k]  = lr0_act[k];  actual_link_cum[0, k]  = lc0_act[k]

    # ----- 초기 과도 완화를 위한 램프/필터 파라미터 -----
    N_ramp = max(1, int(1.5 / dt))   # 약 1.5초 동안 램프
    dq_clip_min, dq_clip_max = 0.4, 1.5
    k_pos_min,  k_pos_max  = 0.3, 2.0
    k_rot_min,  k_rot_max  = 0.3, 2.0
    alpha_min,  alpha_max  = 0.2, 1.0     # q_des 1차 필터 계수
    lam_lp = None
    beta_min,  beta_max  = 0.5, 0.1       # λ 저역통과 계수(초반 진하게→완화)

    # PD 미분항을 위한 이전 에러(선택적 로깅용)
    prev_e_r = np.zeros((3,1)); prev_e_x = np.zeros((3,1))

    # ----- 메인 루프 -----
    for t in range(1, T):
        # 램프 스케일 0→1
        s = min(1.0, t / N_ramp)
        dq_clip_now = dq_clip_min*(1.0 - s) + dq_clip_max*s
        k_pos_now   = k_pos_min *(1.0 - s) + k_pos_max*s
        k_rot_now   = k_rot_min *(1.0 - s) + k_rot_max*s
        alpha_now   = alpha_min *(1.0 - s) + alpha_max*s

        # (1) CLIK로 q_des 업데이트
        T_des = T_des_series[t]
        qd_next, v6, e_r, e_x, J_des = clik_step(
            fk_solver, q_des[t-1][:,0], T_des, dt,
            k_pos=k_pos_now, k_rot=k_rot_now, damp=1e-4,
            dq_clip=dq_clip_now
        )

        # q_des 초반 저역통과(1차 필터)
        qd_col = qd_next.reshape(dof, 1)
        q_des[t]  = alpha_now*qd_col + (1.0 - alpha_now)*q_des[t-1]
        dq_des[t] = ((q_des[t] - q_des[t-1]) / dt)

        # (2) Desired 기록 (EE는 레퍼런스 그대로 저장)
        desired_ee[t] = T_des_series[t]
        lr_des, _ = compute_link_relatives_pure(fk_solver, q_des[t][:,0])
        lc_des    = compute_link_cumulative_from_rel(lr_des)
        for k in range(link_count):
            desired_link_rel[t, k] = lr_des[k]; desired_link_cum[t, k] = lc_des[k]

        # (2-1) EE-PD 렌치(선택) – 필요시 로깅용
        T_now_des = desired_ee[t]
        Rd, pd = T_des[:3,:3], T_des[:3,3]
        Rn, pn = T_now_des[:3,:3], T_now_des[:3,3]
        e_r = orientation_error(Rn, Rd).reshape(3,1)
        e_x = (pd - pn).reshape(3,1)
        de_r = (e_r - prev_e_r) / dt
        de_x = (e_x - prev_e_x) / dt
        prev_e_r, prev_e_x = e_r.copy(), e_x.copy()

        # (3) q→τ (PID)
        cc.set_desired_state({"q": q_des[t], "dq": dq_des[t]})
        tau_joint_cmd = cc.compute_tau(dt, q_act[t-1], dq_act[t-1])
        tau_joint_cmd = clip_inf_norm(tau_joint_cmd, 5e4).reshape(-1)

        # (4) 고장 전 bypass: desired==actual 동기화(학습 안정화)
        if bypass_Blambda and (t0_fault is not None) and (t < t0_fault):
            if t == (t0_fault - 1):
                lam_prev = ext_act.distribute_joint_linf(tau_joint_cmd)  # D^T B λ = τ
            q_act[t]  = q_des[t]
            dq_act[t] = dq_des[t]
            robot.set_joint_states(q_act[t], dq_act[t])
            actual_ee[t] = desired_ee[t]
            actual_link_rel[t] = desired_link_rel[t]
            actual_link_cum[t] = desired_link_cum[t]
            # 고장 전에도 link_wrench[t]는 0으로 두거나, 필요하면 여기서 healthy λ 기반으로 채워도 됨
            continue

        # (5) 분배 λ 계산/적용 (FREEZE MODE + 고장 전 저역통과)
        if (t0_fault is not None) and (t >= t0_fault):
            # 고장 이후: FREEZE 모드
            if not fault_has_triggered:
                if lam_prev is None:
                    lam_prev = ext_act.distribute_joint_linf(tau_joint_cmd)
                lam_frozen = lam_prev.copy()
                if faulty_cols.size > 0:
                    lam_frozen[faulty_cols] = 0.0
                fault_has_triggered = True
            lam_applied = lam_frozen
        else:
            # 고장 전: λ 한 틱 저역통과(초반 진하게)
            lam_cmd = ext_act.distribute_joint_linf(tau_joint_cmd)
            beta_now = beta_min*(1.0 - s) + beta_max*s
            lam_lp = lam_cmd if lam_lp is None else (1.0 - beta_now)*lam_cmd + beta_now*lam_lp
            lam_prev = lam_lp.copy()
            lam_applied = lam_lp

        if fault_has_triggered and faulty_cols.size > 0:
            assert np.all(lam_applied[faulty_cols] == 0.0)

        # (6) 실제 τ 재계산
        D_now = np.asarray(robot.D, dtype=float)
        D_now_s = ext_act._sanitize_mat(D_now, clip=ext_act.MAT_CLIP)
        B_blk_s = ext_act._sanitize_mat(B_blk,  clip=ext_act.MAT_CLIP)
        Aeq_now = D_now_s.T @ B_blk_s
        tau_apply = (Aeq_now @ lam_applied.reshape(-1,1)).reshape(dof,1)

        # ★ 링크별 렌치 계산: w_all = B_blk_s @ λ  → (6L,) 벡터
        w_all = (B_blk_s @ lam_applied.reshape(-1, 1)).reshape(-1)  # (6*link_count,)
        for k in range(link_count):
            start = 6 * k
            end   = 6 * (k + 1)
            link_wrench[t, k, :] = w_all[start:end]

        # (7) 동역학 적분
        nxt = robot.get_next_joint_states(dt, tau_apply)
        qn, dqn = nxt['q'], np.clip(nxt['dq'], -10.0, 10.0)
        if not (_is_finite(qn) and _is_finite(dqn)):
            qn, dqn = q_act[t-1], dq_act[t-1]

        q_act[t], dq_act[t] = qn, dqn
        robot.set_joint_states(qn, dqn)

        # (8) 실제값 로깅
        actual_ee[t]  = fk_solver.compute_end_effector_frame(q_act[t][:,0])
        lr_act, _ = compute_link_relatives_pure(fk_solver, q_act[t][:,0])
        lc_act    = compute_link_cumulative_from_rel(lr_act)
        for k in range(link_count):
            actual_link_rel[t, k]  = lr_act[k]
            actual_link_cum[t, k]  = lc_act[k]

    return (
        desired_ee, actual_ee, label_matrix, onset_vec, first_fault_t, int(t0),
        desired_link_rel, actual_link_rel, desired_link_cum, actual_link_cum,
        dof, np.array(joint_counts, dtype=np.int32),
        link_wrench, which_mask
    )

def _worker(args):
    link_count, T, epsilon_scale, dt, seed, bypass_Blambda = args
    return generate_one_sample(
        link_count=link_count, T=T,
        epsilon_scale=epsilon_scale, dt=dt, seed=seed, bypass_Blambda=bypass_Blambda
    )

def generate_dataset_parallel(link_count, T, NUM_SAMPLES, dt, epsilon_scale, workers, save_dir, timestamps, bypass_Blambda=True):
    if workers is None or workers <= 0:
        workers = os.cpu_count() or 1

    args_list = [(link_count, T, epsilon_scale, dt, 1000+i, bypass_Blambda) for i in range(NUM_SAMPLES)]

    print(f"Spawning {workers} worker(s)...", flush=True)
    done_cnt = 0; start_time = time.time()
    _progress_bar(done_cnt, NUM_SAMPLES, prefix="Generating samples", start_time=start_time)

    # ── 샤드 버퍼 ─────────────────────────────────────────────────────
    shard_idx = _max_existing_shard_index(save_dir)  # 기존 파일 이어붙이기
    print(f"[shard] resume numbering from {shard_idx:05d} (existing max in {save_dir})")
    paths = []

    buf_desired_ee, buf_actual_ee = [], []
    buf_label, buf_onset_vec, buf_first_fault_t, buf_t0 = [], [], [], []
    buf_d_link_rel, buf_a_link_rel = [], []
    buf_d_link_cum, buf_a_link_cum = [], []
    buf_dof, buf_joint_counts = [], []
    buf_link_wrench = []       # ★ (S, T, L, 6)
    buf_fault_mask  = []       # ★ (S, M) = which_mask

    def _flush_shard():
        nonlocal shard_idx, paths
        if not buf_desired_ee:
            return

        desired_ee = _to_f32(np.asarray(buf_desired_ee))        # (S, T, 4, 4)
        actual_ee  = _to_f32(np.asarray(buf_actual_ee))
        label      = np.asarray(buf_label, dtype=np.int32)       # (S, T, M)
        onset_vec  = np.asarray(buf_onset_vec, dtype=np.int32)   # (S, M)
        first_fault_t_arr = np.asarray(buf_first_fault_t, dtype=np.int32)  # (S,)
        t0_arr     = np.asarray(buf_t0, dtype=np.int32)          # (S,)

        d_link_rel = _to_f32(np.asarray(buf_d_link_rel))         # (S, T, L, 4, 4)
        a_link_rel = _to_f32(np.asarray(buf_a_link_rel))
        d_link_cum = _to_f32(np.asarray(buf_d_link_cum))
        a_link_cum = _to_f32(np.asarray(buf_a_link_cum))

        dof_arr          = np.asarray(buf_dof, dtype=np.int32)           # (S,)
        joint_counts_arr = np.asarray(buf_joint_counts, dtype=np.int32)  # (S, L)

        link_wrench_arr = _to_f32(np.asarray(buf_link_wrench))           # (S, T, L, 6)
        fault_mask_arr  = np.asarray(buf_fault_mask, dtype=np.int32)     # (S, M)

        # 다음 번호 배정 + 충돌 시 건너뛰기
        shard_idx += 1
        fname = f"fault_dataset_shard_{shard_idx:05d}.npz"
        path  = os.path.join(save_dir, fname)
        while os.path.exists(path):
            shard_idx += 1
            fname = f"fault_dataset_shard_{shard_idx:05d}.npz"
            path  = os.path.join(save_dir, fname)

        if USE_COMPRESSED:
            np.savez_compressed(
                path,
                desired_link_rel=d_link_rel,
                actual_link_rel=a_link_rel,
                desired_link_cum=d_link_cum,
                actual_link_cum=a_link_cum,
                label=label,
                onset_idx=onset_vec,
                first_fault_t=first_fault_t_arr,
                t0=t0_arr,
                timestamps=timestamps,
                dt=dt,
                link_count=link_count,
                dof=dof_arr,
                joint_counts=joint_counts_arr,
                desired_ee=desired_ee,
                actual_ee=actual_ee,
                link_wrench=link_wrench_arr,
                fault_mask=fault_mask_arr
            )
        else:
            np.savez(
                path,
                desired_link_rel=d_link_rel,
                actual_link_rel=a_link_rel,
                desired_link_cum=d_link_cum,
                actual_link_cum=a_link_cum,
                label=label,
                onset_idx=onset_vec,
                first_fault_t=first_fault_t_arr,
                t0=t0_arr,
                timestamps=timestamps,
                dt=dt,
                link_count=link_count,
                dof=dof_arr,
                joint_counts=joint_counts_arr,
                desired_ee=desired_ee,
                actual_ee=actual_ee,
                link_wrench=link_wrench_arr,
                fault_mask=fault_mask_arr
            )
        print(f"\n[shard] saved: {path}")
        paths.append(path)

        # 버퍼 초기화
        buf_desired_ee.clear(); buf_actual_ee.clear()
        buf_label.clear(); buf_onset_vec.clear(); buf_first_fault_t.clear(); buf_t0.clear()
        buf_d_link_rel.clear(); buf_a_link_rel.clear(); buf_d_link_cum.clear(); buf_a_link_cum.clear()
        buf_dof.clear(); buf_joint_counts.clear()
        buf_link_wrench.clear(); buf_fault_mask.clear()

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        try:
            futs = [ex.submit(_worker, a) for a in args_list]
            for fut in as_completed(futs):
                (d_ee, a_ee, label_mat, onset_vec, first_fault_t, t0,
                 d_lr, a_lr, d_lc, a_lc, dof, joint_counts,
                 link_wrench, fault_mask) = fut.result()

                # 버퍼에 push
                buf_desired_ee.append(_to_f32(d_ee))
                buf_actual_ee.append(_to_f32(a_ee))
                buf_label.append(label_mat.astype(np.int32, copy=False))
                buf_onset_vec.append(onset_vec.astype(np.int32, copy=False))
                buf_first_fault_t.append(int(first_fault_t))
                buf_t0.append(int(t0))

                # (T, L, 4,4) 형태 유지 → (S, T, L, 4,4)
                buf_d_link_rel.append(_to_f32(d_lr))
                buf_a_link_rel.append(_to_f32(a_lr))
                buf_d_link_cum.append(_to_f32(d_lc))
                buf_a_link_cum.append(_to_f32(a_lc))

                buf_dof.append(int(dof))
                buf_joint_counts.append(np.asarray(joint_counts, dtype=np.int32))

                buf_link_wrench.append(_to_f32(link_wrench))
                buf_fault_mask.append(fault_mask.astype(np.int32, copy=False))

                done_cnt += 1
                _progress_bar(done_cnt, NUM_SAMPLES, prefix="Generating samples", start_time=start_time)

                # ✨ 샤드 플러시
                if len(buf_desired_ee) >= SHARD_SIZE:
                    _flush_shard()

        except KeyboardInterrupt:
            ex.shutdown(cancel_futures=True)
            print("\nInterrupted! Flushing remaining buffer...")
            _flush_shard()
            return {"partial": True, "total": done_cnt, "paths": paths}

    # 마지막 잔여 버퍼 flush
    _flush_shard()
    return {"partial": False, "total": done_cnt, "paths": paths}

# ---------------- Main ----------------
if __name__ == '__main__':
    link_count = int(input("How many links?: ").strip())
    try:
        T = int(input("Sequence length T? (default 1000): ").strip())
    except Exception:
        T = 1000
    try:
        NUM_SAMPLES = int(input("How many samples?: ").strip())
    except Exception:
        NUM_SAMPLES = 10
    try:
        workers = int(input("How many workers? (0 = AUTO): ").strip())
    except Exception:
        workers = 0
    if workers == 0:
        workers = os.cpu_count() or 1
    try:
        epsilon_scale = float(input("Epsilon scale (0 for none) [default 0.0]: ").strip())
    except Exception:
        epsilon_scale = 0.0
    try:
        ans = input("Bypass B·λ before first fault? (y/n, default y): ").strip().lower()
        bypass_Blambda = (ans != 'n')
    except Exception:
        bypass_Blambda = True

    save_dir = os.path.join("data_storage", f"link_{link_count}")
    os.makedirs(save_dir, exist_ok=True)

    dt = 0.01
    timestamps = np.arange(T) * dt

    result = generate_dataset_parallel(
        link_count, T, NUM_SAMPLES, dt, epsilon_scale, workers,
        save_dir=save_dir, timestamps=timestamps, bypass_Blambda=bypass_Blambda
    )

    print(f"\nDone. total_samples={result['total']}, partial={result['partial']}")
    print("Shard files:")
    for p in result["paths"]:
        print(" -", p)
    print(f"\nSHARD_SIZE={SHARD_SIZE} | saved in: {save_dir}")
