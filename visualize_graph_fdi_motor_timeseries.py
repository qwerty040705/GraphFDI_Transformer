#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LASDRA Graph-Transformer FDI — Motor-wise time series visualization

- ckpt_epoch_new_080.pth 로 학습된 GraphFDIModel을 불러와서
  shard_00001 ~ shard_00005 사이의 특정 seq_idx 에 대해
  각 모터별(링크 3개 × 모터 8개 = 24개) 시계열을 그림.

- 시계열:
    GT:   label[t, motor] == 0 → fault(1), 아니면 0
    Pred: P(fault on (link ℓ, motor j) at time t) ≈
          P(onset=1) * P(link=ℓ+1) * P(motor=j | link=ℓ)

- 특징/정규화/edge 구성은 graph_fdi_train.py 의
  LASDRAFDIDataset 를 그대로 재사용해서
  훈련 시와 최대한 동일하게 맞춘다.

- 정규화:
    --znorm /path/to/znorm_link3.npz 를 주면,
    그 안의 mu/std 를 사용해 링크 노드 0~8 채널만 정규화한다.
    (LASDRAFDIDataset(normalize=False) + 수동 정규화)
"""

import argparse
import os
from typing import Tuple, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt

# graph_fdi_train.py 가 같은 디렉토리에 있다고 가정
from graph_fdi_train import (
    LASDRAFDIDataset,
    MotorGeom,
    GraphFDIModel,
    setup_device,
)


def resolve_global_seq_idx(
    ds: LASDRAFDIDataset,
    global_seq_idx: int
) -> Tuple[int, int]:
    """
    shard_first~shard_last 로 만든 LASDRAFDIDataset 에서
    '글로벌' 시퀀스 인덱스를 (sh_idx, local_s) 로 변환.

    ds.samples[k]['S'] = 해당 shard 안의 시퀀스 개수
    """
    if global_seq_idx < 0:
        raise IndexError(f"seq_idx must be >= 0 (got {global_seq_idx})")

    remaining = global_seq_idx
    for sh_idx, pack in enumerate(ds.samples):
        S = pack["S"]
        if remaining < S:
            return sh_idx, remaining
        remaining -= S

    total_S = sum(p["S"] for p in ds.samples)
    raise IndexError(
        f"seq_idx {global_seq_idx} is out of range. "
        f"Total sequences over all shards = {total_S} (0 ~ {total_S-1})"
    )


def build_model_from_ckpt(ckpt_path: str, device: torch.device,
                          links: int, motors_per_link: int = 8) -> GraphFDIModel:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg_dict = ckpt.get("cfg", {})

    d_model = cfg_dict.get("d_model", 384)
    heads = cfg_dict.get("heads", 6)
    depth = cfg_dict.get("depth", 5)
    dropout = cfg_dict.get("dropout", 0.1)
    temporal = cfg_dict.get("temporal", "transformer")

    d_node_in = 10   # link 9D + motor_sim 1D (훈련 스크립트와 동일)
    d_edge_in = 16   # [link-link 6 | link-motor 10]

    model = GraphFDIModel(
        d_node_in=d_node_in,
        d_edge_in=d_edge_in,
        d_model=d_model,
        heads=heads,
        depth=depth,
        temporal=temporal,
        d_temporal=d_model,
        links=links,
        motors_per_link=motors_per_link,
        dropout=dropout,
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[INFO] Loaded checkpoint from: {ckpt_path}")
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Visualize per-motor time series (GT vs predicted) for Graph-FDI."
    )
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to ckpt_epoch_new_XXX.pth")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to link_3 directory containing fault_dataset_shard_*.npz")
    parser.add_argument("--shard_first", type=int, default=1)
    parser.add_argument("--shard_last", type=int, default=5)
    parser.add_argument("--seq_idx", type=int, default=0,
                        help="Global sequence index over [shard_first, shard_last]")
    parser.add_argument("--motor_geom", type=str, default=None,
                        help="Path to motor_geom.npz (same as training)")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out", type=str, default="graph_fdi_motor_timeseries.png",
                        help="Output PNG filename")

    # ★ precomputed z-norm (mu/std) 경로
    parser.add_argument("--znorm", type=str, default=None,
                        help="Path to znorm_link3.npz (must contain 'mu' and 'std')")

    args = parser.parse_args()

    # ----------------- Device -----------------
    device = setup_device(args.device)

    # ----------------- z-norm (mu/std) 로드 -----------------
    mu: Optional[np.ndarray] = None
    std: Optional[np.ndarray] = None
    if args.znorm is not None:
        print(f"[INFO] Loading z-norm from: {args.znorm}")
        zn = np.load(args.znorm)
        mu = zn["mu"].astype(np.float32)   # (9,)
        std = zn["std"].astype(np.float32) # (9,)
        print(f"[INFO] Loaded mu/std with shape: mu={mu.shape}, std={std.shape}")
    else:
        print("[WARN] --znorm not provided. Running WITHOUT normalization (this may hurt performance).")

    # ----------------- Dataset (for features + cache) -----------------
    shard_ids = list(range(args.shard_first, args.shard_last + 1))
    print(f"[INFO] Using shards: {shard_ids}")

    geom = MotorGeom(args.motor_geom) if args.motor_geom else None

    # 여기서는 normalize=False 로 두고, 위에서 로드한 mu/std 로 직접 정규화한다.
    full_ds = LASDRAFDIDataset(
        data_dir=args.data_dir,
        shard_ids=shard_ids,
        T_win=128,
        stride=64,
        motors_per_link=8,
        normalize=False,
        motor_geom=geom,
        cache_windows=True,
        cache_dir="./cache_L3",
        cache_dtype="fp32",
    )

    L = full_ds.L             # 링크 개수 (보통 3)
    mpl = full_ds.mpl         # 모터/링크 (8)
    M_total = L * mpl

    print(f"[INFO] link_count = {L}, motors_per_link = {mpl} → total motors = {M_total}")

    # ----------------- Seq index 해석 -----------------
    sh_idx, local_s = resolve_global_seq_idx(full_ds, args.seq_idx)
    pack = full_ds.samples[sh_idx]
    sid = pack["sid"]
    S = pack["S"]
    T_all = pack["T"]
    dt = pack.get("dt", 1.0)

    print(f"[INFO] Global seq_idx {args.seq_idx} → shard_id={sid} (sh_idx={sh_idx}), local seq={local_s}")
    print(f"[INFO] This sequence length T = {T_all}, dt = {dt}")

    # ----------------- GT motor fault 시계열 -----------------
    # label: (S, T, 8L), 0=fault, 1=normal
    label_full = pack["label"][local_s]      # (T_all, 8L)
    if label_full.shape[1] != M_total:
        raise RuntimeError(
            f"label dim mismatch: got {label_full.shape[1]}, expected {M_total}"
        )

    # gt_fault[t, m] ∈ {0,1}
    gt_fault = np.zeros((T_all, M_total), dtype=np.float32)
    gt_fault[label_full == 0] = 1.0  # 0일 때 fault → 1로 바꿈

    # ----------------- 모델 로드 -----------------
    model = build_model_from_ckpt(args.ckpt, device, links=L, motors_per_link=mpl)

    # ----------------- Pred 시계열 배열 -----------------
    # pred_fault[t, m] ∈ [0,1], 초기값 NaN
    pred_fault = np.full((T_all, M_total), np.nan, dtype=np.float32)

    T_win = full_ds.T_win  # 128

    # ----------------- Sliding window 로 시간 t별 예측 -----------------
    model.eval()
    with torch.no_grad():
        for t_cur in range(T_all):
            # 윈도우가 완전히 만들어지는 시점부터 예측 가능 (t_cur >= T_win-1)
            if t_cur < T_win - 1:
                continue

            t0 = t_cur - T_win + 1  # 윈도우 시작 시점

            # LASDRAFDIDataset._extract_window를 직접 호출해서
            # 훈련과 동일한 feature / edge / mask 구성 사용
            node_feat, edge_feat, attn_mask, _ = full_ds._extract_window(
                sh_idx, local_s, t0
            )  # node_feat: (T_win, N, 10), edge_feat: (T_win, N, N, 16)

            # 링크 노드 0~8 채널만 정규화 (mu/std 가 주어졌을 때만)
            if (mu is not None) and (std is not None):
                L_links = full_ds.L
                node_feat[:, :L_links, 0:9] = (
                    (node_feat[:, :L_links, 0:9] - mu) / std
                )

            node_feat = np.nan_to_num(node_feat, nan=0.0, posinf=1e6, neginf=-1e6)
            edge_feat = np.nan_to_num(edge_feat, nan=0.0, posinf=1e6, neginf=-1e6)

            X = torch.from_numpy(node_feat).float().unsqueeze(0).to(device)   # (1,T,N,D)
            E = torch.from_numpy(edge_feat).float().unsqueeze(0).to(device)   # (1,T,N,N,De)
            M_mask = torch.from_numpy(attn_mask).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,N,N)

            o1, o2, m_logits = model(X, E, M_mask)  # o1: (1,), o2: (1,L+1), m_logits: list[L] of (1, mpl)

            # ----------------- 확률로 변환 -----------------
            p_onset = torch.sigmoid(o1[0]).item()            # P(onset=1)
            link_probs = torch.softmax(o2[0], dim=-1)        # len L+1 (0=normal, 1..L=fault link)

            # 각 링크별 모터 분포 P(motor=j | link=ℓ)
            motor_probs_links = []
            for li in range(L):
                # m_logits[li]: (1, mpl)
                probs_li = torch.softmax(m_logits[li][0], dim=-1)  # (mpl,)
                motor_probs_links.append(probs_li.cpu().numpy())   # (mpl,)

            # motor index: m = li * mpl + mj
            for li in range(L):
                p_link_li = float(link_probs[li + 1].item())  # link_class=li+1
                for mj in range(mpl):
                    p_mj_given_li = float(motor_probs_links[li][mj])
                    m_idx = li * mpl + mj

                    # P(fault at (li, mj)) ≈ P(onset) * P(link=li+1) * P(motor=j | link)
                    pred_fault[t_cur, m_idx] = (
                        p_onset * p_link_li * p_mj_given_li
                    )

    # ----------------- Plot -----------------
    time_axis = np.arange(T_all) * dt

    fig, axes = plt.subplots(
        nrows=L,
        ncols=mpl,
        figsize=(mpl * 3.0, L * 2.5),
        sharex=True,
        sharey=True,
    )

    if L == 1:
        axes = np.expand_dims(axes, axis=0)  # (1, mpl)
    if mpl == 1:
        axes = np.expand_dims(axes, axis=1)  # (L, 1)

    for li in range(L):
        for mj in range(mpl):
            ax = axes[li, mj]
            m_idx = li * mpl + mj

            gt = gt_fault[:, m_idx]
            pred = pred_fault[:, m_idx]

            # 예측 시계열
            ax.plot(time_axis, pred, label="Pred", linewidth=1.2)

            # GT (0/1) 는 step 형식으로 그려줌
            ax.step(time_axis, gt, where="post", linestyle="--", linewidth=0.8, label="GT")

            ax.set_ylim(-0.05, 1.05)
            if li == L - 1:
                ax.set_xlabel("t [s]")
            if mj == 0:
                ax.set_ylabel("P(fault)")

            ax.set_title(f"L{li+1}-M{mj}", fontsize=9)

            # 첫 줄 첫 칸에만 legend
            if li == 0 and mj == 0:
                ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"[INFO] Saved figure to: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()

"""
사용 예시
--------

python3 visualize_graph_fdi_motor_timeseries.py \
  --ckpt /home/user/transformer_fault_diagnosis/GraphFDI_Transformer_L3/GraphFDI_Transformer_L3.pth \
  --data_dir /home/user/transformer_fault_diagnosis/data_storage/link_3 \
  --znorm /home/user/transformer_fault_diagnosis/GraphFDI_Transformer_L3/znorm_link3.npz \
  --shard_first 1 --shard_last 6 \
  --seq_idx 5000 \
  --motor_geom /home/user/transformer_fault_diagnosis/motor_geom.npz \
  --out motor_timeseries_seq5000.png
"""
