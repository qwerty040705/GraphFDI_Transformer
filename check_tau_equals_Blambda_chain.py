#!/usr/bin/env python3
"""
통합 검증: tau ?= D^T · B_blkdiag · lambda  (논문식 nullspace + l_inf 분배기)

실행 예:
  python3 check_tau_equals_Blambda_chain.py  --links 3 --trials 50

출력:
  - 각 trial별 ||tau_hat - tau||_inf
  - 전체 통계 (max/mean), nullspace 사용 차원/포화치 비율
"""

import os
import sys
import argparse
import numpy as np

# ── 프로젝트 경로 합류 ───────────────────────────────────────────────
ROOT = os.path.abspath(os.path.dirname(__file__))  # 이 파일이 control/ 밑이 아니면 알맞게 조정
sys.path.append(ROOT)
sys.path.append(os.path.abspath(os.path.join(ROOT, "..")))
sys.path.append(os.path.abspath(os.path.join(ROOT, "..", "dynamics")))
sys.path.append(os.path.abspath(os.path.join(ROOT, "..", "data_generate")))

from parameters import get_parameters
from parameters_model import parameters_model
from dynamics.lasdra_class import LASDRA

# 외부 분배기 (논문식 per-link l∞)
from control.external_actuation import ExternalActuation


def build_B_blkdiag(params_model):
    # D는 [τ; f] 순서, B는 [f; τ] → swap해서 정렬
    eye_perm = np.block([
        [np.zeros((3,3)), np.eye(3)],
        [np.eye(3), np.zeros((3,3))]
    ])
    B_blocks = [eye_perm @ np.asarray(odar.B, dtype=float) for odar in params_model["ODAR"]]
    # scipy.sparse.block_diag 대신 간단 블록대각
    m_total = sum(B.shape[1] for B in B_blocks)
    Bbd = np.zeros((6*len(B_blocks), m_total), dtype=float)
    col = 0
    for i, Bi in enumerate(B_blocks):
        mi = Bi.shape[1]
        Bbd[6*i:6*(i+1), col:col+mi] = Bi
        col += mi
    return Bbd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--links", type=int, default=3, help="number of links")
    ap.add_argument("--trials", type=int, default=20, help="random trials")
    ap.add_argument("--seed", type=int, default=42, help="random seed")
    ap.add_argument("--tau_scale", type=float, default=200.0, help="random tau magnitude")
    args = ap.parse_args()

    np.random.seed(args.seed)

    # ── 파라미터 & 모델 구성 (LASDRA with 'links' links) ────────────
    base_param = get_parameters(args.links)
    base_param["ODAR"] = base_param["ODAR"][:args.links]

    screw_axes, inertias = [], []
    for odar in base_param["ODAR"]:
        screw_axes.extend(odar.body_joint_screw_axes)
        inertias.extend(odar.joint_inertia_tensor)
    base_param["LASDRA"].update(
        body_joint_screw_axes=screw_axes,
        inertia_matrix=inertias,
        dof=len(screw_axes),
    )
    model_param = parameters_model(mode=0, params_prev=base_param)
    robot = LASDRA(model_param)

    dof = model_param["LASDRA"]["dof"]
    B_blk = build_B_blkdiag(model_param)

    # 논문식 분배기
    ext = ExternalActuation(model_param, robot)

    # ── 랜덤 상태/토크에서 체인 검사 ────────────────────────────────
    resids = []
    sat_counts = []
    used_ns_dims = []

    for t in range(args.trials):
        # 랜덤 q, dq 설정 (범위는 적당히)
        q  = (np.random.rand(dof, 1) * 2 - 1) * 0.2 * np.pi
        dq = (np.random.randn(dof, 1)) * 0.0
        robot.set_joint_states(q, dq)

        # 현재 D, 그리고 블록대각 B
        D = np.asarray(robot.D, dtype=float)  # (6L×d)
        A = D.T @ B_blk                        # (d×8L)

        # 주의: 임의 τ가 항상 A의 열공간에 있지 않음.
        # 논문식 분배기 내부는 D^T F = τ를 만족하는 F를 만들고, per-link에서 Bpλ = Fi를 nullspace로 유지.
        # 따라서 Aλ = τ가 정확히 나와야 함.
        tau = (np.random.randn(dof, 1)).reshape(-1, 1)
        tau *= (args.tau_scale / max(np.linalg.norm(tau, ord=np.inf), 1e-9))

        # λ 산출 (논문식: per-link nullspace + l_inf)
        lam = ext.distribute_torque_nullspace_inf(tau.reshape(-1))

        # 재구성 τ̂
        tau_hat = (A @ lam.reshape(-1,1)).reshape(-1)

        resid = float(np.linalg.norm(tau_hat.reshape(-1,1) - tau, ord=np.inf))
        resids.append(resid)

        # 통계(선택): 외부 분배기 내부 통계는 반환치가 없으므로 간접 측정
        # 여기서는 λ의 박스 경계 포화 개수를 기록
        lb = ext.qp["lb"] if "lb" in ext.qp else -np.inf*np.ones_like(lam)
        ub = ext.qp["ub"] if "ub" in ext.qp else  np.inf*np.ones_like(lam)
        sat = int(np.sum((lam <= lb + 1e-9) | (lam >= ub - 1e-9)))
        sat_counts.append(sat)

        # per-link nullspace 총 차원 (참고용)
        ns_total = sum(al.N.shape[1] for al in ext.allocs)
        used_ns_dims.append(ns_total)

        print(f"[{t+1:02d}/{args.trials}] ||Aλ-τ||_inf = {resid:.3e}   "
              f"(sat={sat}, ns_dim_total={ns_total})")

    resids = np.array(resids)
    print("\n==== Summary ====")
    print(f"trials   : {args.trials}")
    print(f"max resid: {resids.max():.6e}")
    print(f"mean resid: {resids.mean():.6e}")
    print(f"median resid: {np.median(resids):.6e}")
    print(f"mean saturated λ count: {np.mean(sat_counts):.2f}")
    print(f"nullspace dims (per-link sum): unique {sorted(set(used_ns_dims))}")


if __name__ == "__main__":
    main()
