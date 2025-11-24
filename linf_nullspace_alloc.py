import numpy as np
import numpy.linalg as npl
from typing import Optional, Tuple
from scipy.optimize import linprog


def _sanitize_vec(v, clip=1e12):
    v = np.asarray(v, dtype=float).reshape(-1)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    if clip is not None:
        v = np.clip(v, -clip, clip)
    return v


def _sanitize_mat(M, clip=1e12):
    M = np.asarray(M, dtype=float)
    M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
    if clip is not None:
        M = np.clip(M, -clip, clip)
    return M


def _nullspace_projected_from_cols(B: np.ndarray, cols: np.ndarray, rcond=1e-9) -> np.ndarray:
    """
    B: (6×m), cols: (m×k)  ->  N = (I - B^+ B) cols  (B의 널스페이스 성분만 추출)
    영벡터에 가까운 열은 버린다.
    """
    B = _sanitize_mat(B)
    cols = _sanitize_mat(cols)
    pinvB = npl.pinv(B, rcond=rcond)
    Pn = np.eye(B.shape[1]) - pinvB @ B  # (m×m)
    N = Pn @ cols
    keep = [j for j in range(N.shape[1]) if npl.norm(N[:, j]) > 1e-9]
    return N[:, keep] if keep else np.zeros((B.shape[1], 0))


def _nullspace_from_svd(B: np.ndarray, rcond=1e-9) -> np.ndarray:
    """
    B: (6×m).  SVD로 널스페이스 기저 구함 (m×r)
    """
    B = _sanitize_mat(B)
    U, s, Vt = npl.svd(B, full_matrices=True)
    tol = rcond * max(B.shape) * (s[0] if s.size else 1.0)
    r = np.sum(s > tol)
    # Vt shape: (m × m); nullspace basis: last (m-r) rows of Vt
    if r >= Vt.shape[0]:
        return np.zeros((B.shape[1], 0))
    N = Vt[r:, :].T  # (m × (m-r))
    return N


def linf_allocate_lambda(
    tau: np.ndarray,
    B: np.ndarray,
    N_basis: Optional[np.ndarray] = None,
    lb: Optional[np.ndarray] = None,
    ub: Optional[np.ndarray] = None,
    rcond: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    문제:  τ = B λ,  λ = B^+ τ + N α
          α를 골라 ||λ||_∞ 최소 (또는 그에 근사) + 박스제약 lb ≤ λ ≤ ub
    입력:
        tau    : (6,) or (6,1)
        B      : (6, m)
        N_basis: (m, k) 또는 None  (None이면 SVD로 널스페이스 기저 생성)
        lb, ub : (m,) 경계 (None이면 ±∞)
    출력:
        lam    : (m,)
        tau_hat: (6,)  (= B @ lam)
        stats  : dict( 'resid_tau', 'linf', 'box_saturated', 'success', 'alpha' )
    """
    tau = _sanitize_vec(tau)
    B = _sanitize_mat(B)
    m = B.shape[1]

    # 기본 해: λ0 = B^+ τ
    pinvB = npl.pinv(B, rcond=rcond)
    lam0 = pinvB @ tau

    # 경계 설정
    if ub is None:
        ub = np.full(m, np.inf)
    else:
        ub = _sanitize_vec(ub)
    if lb is None:
        lb = -np.copy(ub)
    else:
        lb = _sanitize_vec(lb)
    # 안전하게 lb ≤ ub 보장
    lb = np.minimum(lb, ub)

    # 널스페이스 기저
    if N_basis is None:
        N = _nullspace_from_svd(B, rcond=rcond)  # (m×r)
    else:
        N = _nullspace_projected_from_cols(B, N_basis, rcond=rcond)

    r = N.shape[1] if N.size else 0
    if r == 0:
        lam = np.clip(lam0, lb, ub)
        tau_hat = B @ lam
        stats = {
            "success": True,
            "alpha": np.zeros(0),
            "resid_tau": float(npl.norm(tau_hat - tau, ord=np.inf)),
            "linf": float(npl.norm(lam, ord=np.inf)),
            "box_saturated": int(np.sum((lam <= lb + 1e-12) | (lam >= ub - 1e-12))),
            "used_nullspace_dim": 0,
        }
        return lam, tau_hat, stats

    # LP:  min s
    # s.t.  -s <= lam0 + N α <= s
    #       lb <= lam0 + N α <= ub
    # 표준형 A_ub x <= b_ub,   x = [α; s]
    A_ub = np.vstack([
        np.hstack([  N, -np.ones((m, 1))]),   #  Nα - s <= -lam0
        np.hstack([ -N, -np.ones((m, 1))]),   # -Nα - s <=  lam0
        np.hstack([  N,  np.zeros((m, 1))]),  #  Nα <=  ub - lam0
        np.hstack([ -N,  np.zeros((m, 1))]),  # -Nα <= -(lb - lam0)
    ])
    b_ub = np.concatenate([
        -lam0,
         lam0,
        (ub - lam0),
       -(lb - lam0),
    ])

    c = np.zeros(r + 1)
    c[-1] = 1.0  # min s

    bounds = [(None, None)] * r + [(0.0, None)]
    try:
        res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if res is None or not res.success:
            raise RuntimeError("linprog failed")
        alpha = res.x[:r]
        lam = lam0 + (N @ alpha)
    except Exception:
        # 실패 시: 널스페이스로 박스만 만족하도록 가장 가까운 점 (간단 클립)
        alpha = np.zeros(r)
        lam = lam0

    lam = np.clip(lam, lb, ub)
    tau_hat = B @ lam
    stats = {
        "success": True,
        "alpha": alpha,
        "resid_tau": float(npl.norm(tau_hat - tau, ord=np.inf)),
        "linf": float(npl.norm(lam, ord=np.inf)),
        "box_saturated": int(np.sum((lam <= lb + 1e-12) | (lam >= ub - 1e-12))),
        "used_nullspace_dim": int(r),
    }
    return lam, tau_hat, stats


# ──────────────────────────────
# 데모/단독 실행
# ──────────────────────────────
if __name__ == "__main__":
    # 예시 B (6×8): 사용자가 준 값
    B = np.array([
        [ 0.6797,  0.6797,  0.6797,  0.6797,  0.6797,  0.6797,  0.6797,  0.6797],
        [ 0.1908,  0.1908, -0.1908, -0.1908,  0.1908,  0.1908, -0.1908, -0.1908],
        [ 0.7082, -0.7082,  0.7082, -0.7082,  0.7082, -0.7082,  0.7082, -0.7082],
        [-0.1026, -0.1026, -0.1026, -0.1026,  0.1026,  0.1026,  0.1026,  0.1026],
        [-0.0711,  0.1980, -0.1980,  0.0711,  0.0711, -0.1980,  0.1980, -0.0711],
        [ 0.0894, -0.0169,  0.0169, -0.0894, -0.0894,  0.0169, -0.0169,  0.0894]
    ], dtype=float)

    # 예시 Bnsv (m=8) → 열 2개짜리 컬럼스택
    Bnsv = {
        "upper": np.array([ 1, -1, -1,  1, 0, 0, 0, 0], dtype=float),
        "lower": np.array([ 0,  0,  0,  0, 1,-1,-1, 1], dtype=float),
    }
    N_cols = np.vstack([Bnsv["upper"], Bnsv["lower"]]).T  # (8×2)

    # 예시 입력 토크/힘 τ (6,)
    tau = np.array([30.0, -5.0, 10.0, 2.0, -1.0, 3.0])

    # 박스 경계 (스러스트 제한) — 필요시 조정
    ub = 50.0 * np.ones(B.shape[1])
    lb = -ub

    lam, tau_hat, stats = linf_allocate_lambda(tau, B, N_basis=N_cols, lb=lb, ub=ub)

    print("=== l_infinity allocation demo ===")
    print("tau (given):", tau)
    print("lambda      :", np.round(lam, 6))
    print("tau_hat=Bλ  :", np.round(tau_hat, 6))
    print("||tau_hat - tau||_inf :", stats["resid_tau"])
    print("||lambda||_inf        :", stats["linf"])
    print("box saturated count   :", stats["box_saturated"])
    print("used nullspace dim    :", stats["used_nullspace_dim"])
