import os
import math
import argparse
import random
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F 

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import proj3d  

from graph_fdi_train import (
    LASDRAFDIDataset,
    MotorGeom,
    GraphFDIModel,
    setup_device,
)


# ============================================================
#                    Utilities / helpers
# ============================================================

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_global_seq_idx(
    ds: LASDRAFDIDataset,
    global_seq_idx: int
) -> Tuple[int, int]:
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


def build_model_from_ckpt(
    ckpt_path: str,
    device: torch.device,
    links: int,
    motors_per_link: int = 8
) -> Tuple[GraphFDIModel, dict]:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg_dict = ckpt.get("cfg", {})

    d_model = cfg_dict.get("d_model", 384)
    heads = cfg_dict.get("heads", 6)
    depth = cfg_dict.get("depth", 5)
    dropout = cfg_dict.get("dropout", 0.1)
    temporal = cfg_dict.get("temporal", "transformer")

    d_node_in = 10   # link 9D + motor_sim 1D
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
    return model, cfg_dict


def add_base_pose(cum_seq: np.ndarray) -> np.ndarray:
    T, L = cum_seq.shape[:2]
    out = np.zeros((T, L + 1, 4, 4), dtype=cum_seq.dtype)
    out[:, 0] = np.eye(4, dtype=cum_seq.dtype)[None, ...]
    out[:, 1:] = cum_seq
    return out


def compute_pred_fault_series(
    ds: LASDRAFDIDataset,
    sh_idx: int,
    local_s: int,
    model: GraphFDIModel,
    device: torch.device,
    mu: Optional[np.ndarray],
    std: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    pack = ds.samples[sh_idx]
    T_all = pack["T"]
    L = ds.L
    mpl = ds.mpl
    M_total = L * mpl

    T_win = ds.T_win

    if mu is None or std is None:
        ds_mu = getattr(ds, "_mu", None)
        ds_std = getattr(ds, "_std", None)
        if ds_mu is not None and ds_std is not None:
            print("[WARN] Using ds._mu/_std (znorm not provided).")
            mu = ds_mu
            std = ds_std
        else:
            print("[WARN] No normalization (mu/std not provided and ds._mu is None).")

    pred_fault = np.full((T_all, M_total), np.nan, dtype=np.float32)
    onset_prob = np.zeros(T_all, dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for t_cur in range(T_all):
            if t_cur < T_win - 1:
                continue

            t0 = t_cur - T_win + 1

            node_feat, edge_feat, attn_mask, _ = ds._extract_window(sh_idx, local_s, t0)

            # 링크 노드 0~8 채널만 정규화 (mu/std 가 있을 때만)
            if mu is not None and std is not None:
                node_feat[:, :L, 0:9] = (node_feat[:, :L, 0:9] - mu) / std

            node_feat = np.nan_to_num(node_feat, nan=0.0, posinf=1e6, neginf=-1e6)
            edge_feat = np.nan_to_num(edge_feat, nan=0.0, posinf=1e6, neginf=-1e6)

            X = torch.from_numpy(node_feat).float().unsqueeze(0).to(device)      # (1,T,N,D)
            E = torch.from_numpy(edge_feat).float().unsqueeze(0).to(device)      # (1,T,N,N,De)
            M_mask = torch.from_numpy(attn_mask).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,N,N)

            o1, o2, m_logits = model(X, E, M_mask)  # o1: (1,), o2: (1,L+1), m_logits: list[L] of (1,mpl)

            p_onset = torch.sigmoid(o1[0]).item()
            onset_prob[t_cur] = p_onset

            link_probs = torch.softmax(o2[0], dim=-1)  # (L+1,)

            motor_probs_links = []
            for li in range(L):
                probs_li = torch.softmax(m_logits[li][0], dim=-1)  # (mpl,)
                motor_probs_links.append(probs_li.cpu().numpy())

            for li in range(L):
                p_link_li = float(link_probs[li + 1].item())  # link_class = li+1
                for mj in range(mpl):
                    p_mj_given_li = float(motor_probs_links[li][mj])
                    m_idx = li * mpl + mj
                    pred_fault[t_cur, m_idx] = (
                        p_onset * p_link_li * p_mj_given_li
                    )

    pred_fault = np.nan_to_num(pred_fault, nan=0.0, posinf=1.0, neginf=0.0)
    return pred_fault, onset_prob


def kofn_top1_smoothing(
    prob_motor: np.ndarray,
    theta: float = 0.8,
    K: int = 8,
    N: int = 10,
) -> np.ndarray:
    T, M = prob_motor.shape
    top1_prob = np.max(prob_motor, axis=-1)   # (T,)
    top1_idx = np.argmax(prob_motor, axis=-1) # (T,)

    out = np.zeros(T, dtype=np.int32)

    for t in range(T):
        m_cur = top1_idx[t]
        t0 = max(0, t - N + 1)
        mask = (top1_idx[t0:t+1] == m_cur) & (top1_prob[t0:t+1] >= theta)
        if mask.sum() >= K:
            out[t] = m_cur + 1  # 1..M
        else:
            out[t] = 0
    return out


def _norm(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + eps)


# ============================================================
#                           Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="LASDRA Graph-Transformer FDI streaming visualizer (GraphFDIModel)."
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

    # ★ precomputed z-norm (mu/std) 경로
    parser.add_argument("--znorm", type=str, default=None,
                        help="Path to znorm_link3.npz (must contain 'mu' and 'std')")

    # playback / world
    parser.add_argument("--speed", type=float, default=1.0,
                        help=">1.0 이면 실제 시간보다 빠르게 재생")
    parser.add_argument("--data_hz_override", type=float, default=-1.0,
                        help=">0 이면 shard dt 대신 이 값을 사용 (Hz)")
    parser.add_argument("--fix_origin", type=int, default=1,
                        help="1이면 BASE 위치를 항상 (0,0,0) 으로 정렬")

    # motors/layout
    parser.add_argument("--motors_per_link", type=int, default=8)
    parser.add_argument("--anchor_ratio", type=float, default=0.85)
    parser.add_argument("--arm_len", type=float, default=0.22)

    # props
    parser.add_argument("--prop_blades", type=int, default=4)
    parser.add_argument("--prop_radius", type=float, default=0.10)
    parser.add_argument("--prop_chord",  type=float, default=0.035)
    parser.add_argument("--prop_alpha",  type=float, default=0.85)
    parser.add_argument("--stem_alpha",  type=float, default=0.95)
    parser.add_argument("--prop_rps", type=float, default=15.0)
    parser.add_argument("--spin_dir_alt", type=int, default=1,
                        help="1이면 front/back/짝홀 motor 별로 회전 방향 교차")

    # K-of-N (non-latch)
    parser.add_argument("--theta_motor", type=float, default=0.8,
                        help="모터 fault 로 볼 top-1 prob threshold (기본 0.8)")
    parser.add_argument("--kofn_k", type=int, default=8,
                        help="K-of-N smoothing 의 K (기본 8)")
    parser.add_argument("--kofn_n", type=int, default=10,
                        help="K-of-N smoothing 의 N (기본 10)")

    # 3D view rotation
    parser.add_argument("--rotate_speed", type=float, default=30.0,
                        help="3D view azimuth 회전 속도 [deg/sec]")

    # video
    parser.add_argument("--save_video", type=int, default=0)
    parser.add_argument("--out", type=str, default="lasdra_vis.mp4")
    parser.add_argument("--video_fps", type=int, default=100)
    parser.add_argument("--codec", type=str, default="libx264")
    parser.add_argument("--bitrate", type=str, default="4000k")
    parser.add_argument("--dpi", type=int, default=150)

    parser.add_argument("--cache_dir", type=str, default="./cache_L3")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)
    device = setup_device(args.device)
    torch.set_float32_matmul_precision("high")

    # ----------------- z-norm (mu/std) 로드 -----------------
    mu = None
    std = None
    if args.znorm is not None:
        print(f"[INFO] Loading z-norm from: {args.znorm}")
        zn = np.load(args.znorm)
        mu = zn["mu"].astype(np.float32)
        std = zn["std"].astype(np.float32)
        print(f"[INFO] Loaded mu/std with shape: mu={mu.shape}, std={std.shape}")
    else:
        print("[WARN] --znorm not provided. Will fallback to ds._mu/_std if available.")

    # ----------------- Dataset -----------------
    shard_ids = list(range(args.shard_first, args.shard_last + 1))
    print(f"[INFO] Using shards: {shard_ids}")

    geom = MotorGeom(args.motor_geom) if args.motor_geom else None

    ckpt_tmp = torch.load(args.ckpt, map_location="cpu")
    cfg_ckpt = ckpt_tmp.get("cfg", {})

    T_win = int(cfg_ckpt.get("T_win", 128))
    stride = int(cfg_ckpt.get("stride", 64))
    cache_windows = int(cfg_ckpt.get("cache_windows", 1))
    cache_dtype = str(cfg_ckpt.get("cache_dtype", "fp32"))
    cache_dir = cfg_ckpt.get("cache_dir", args.cache_dir)


    full_ds = LASDRAFDIDataset(
        data_dir=args.data_dir,
        shard_ids=shard_ids,
        T_win=T_win,
        stride=stride,
        motors_per_link=args.motors_per_link,
        normalize=False,
        motor_geom=geom,
        cache_windows=bool(cache_windows),
        cache_dir=cache_dir,
        cache_dtype=cache_dtype,
    )

    L = full_ds.L
    mpl = full_ds.mpl
    M_total = L * mpl

    print(f"[INFO] link_count = {L}, motors_per_link = {mpl} → total motors = {M_total}")

    # ----------------- Seq index / basic arrays -----------------
    sh_idx, local_s = resolve_global_seq_idx(full_ds, args.seq_idx)
    pack = full_ds.samples[sh_idx]
    sid = pack["sid"]
    S = pack["S"]
    T_all = pack["T"]
    dt = float(pack.get("dt", 0.01))

    print(f"[INFO] Global seq_idx {args.seq_idx} → shard_id={sid} (sh_idx={sh_idx}), local seq={local_s}")
    print(f"[INFO] This sequence length T = {T_all}, dt = {dt}")

    data_hz = args.data_hz_override if args.data_hz_override > 0 else (1.0 / max(dt, 1e-9))
    print(f"[INFO] data_hz = {data_hz:.3f} Hz, playback speed = {args.speed:.2f}x")

    desired_cum_seq = pack["desired_link_cum"][local_s]  # (T,L,4,4)
    actual_cum_seq = pack["actual_link_cum"][local_s]    # (T,L,4,4)

    Dcum = add_base_pose(desired_cum_seq)  # (T,L+1,4,4)
    Acum = add_base_pose(actual_cum_seq)   # (T,L+1,4,4)

    labels_seq = pack["label"][local_s]    # (T, 8L)
    if labels_seq.shape[1] != M_total:
        raise RuntimeError(
            f"label dim mismatch: got {labels_seq.shape[1]}, expected {M_total}"
        )
    gt_fault = np.zeros((T_all, M_total), dtype=np.float32)
    gt_fault[labels_seq == 0] = 1.0  # 0=fault → 1 로 변환

    # ----------------- Model & predictions -----------------
    model, _ = build_model_from_ckpt(args.ckpt, device, links=L, motors_per_link=mpl)
    pred_fault, onset_prob = compute_pred_fault_series(
        full_ds, sh_idx, local_s, model, device, mu, std
    )

    # K-of-N top1 smoothing (non-latch)
    kofn_idx = kofn_top1_smoothing(
        pred_fault,
        theta=args.theta_motor,
        K=args.kofn_k,
        N=args.kofn_n,
    )  # (T,), 0=BG, 1..M

    time_axis = np.arange(T_all) * dt

    # ----------------- GT / Pred 고장 시점 및 지연 계산 -----------------
    gt_any = gt_fault >= 0.5
    gt_any_any = gt_any.any(axis=1)
    if gt_any_any.any():
        t_fault_gt = int(np.where(gt_any_any)[0][0])
        m_fault_gt = int(np.where(gt_any[t_fault_gt])[0][0])
        link_gt = m_fault_gt // mpl + 1
        motor_gt = m_fault_gt % mpl + 1
    else:
        t_fault_gt = None
        m_fault_gt = None
        link_gt = None
        motor_gt = None

    pred_any = kofn_idx > 0
    if pred_any.any():
        t_fault_pred = int(np.where(pred_any)[0][0])
        m_fault_pred = int(kofn_idx[t_fault_pred] - 1)
        link_pred = m_fault_pred // mpl + 1
        motor_pred = m_fault_pred % mpl + 1
    else:
        t_fault_pred = None
        m_fault_pred = None
        link_pred = None
        motor_pred = None

    if (t_fault_gt is not None) and (t_fault_pred is not None):
        delay_s = (t_fault_pred - t_fault_gt) * dt
    else:
        delay_s = None

    # =======================================================
    #               Matplotlib figure & axes
    # =======================================================
    rows = L
    fig_h = 6 + 1.8 * max(0, rows - 1)
    plt.close("all")
    fig = plt.figure(figsize=(12, fig_h))
    gs = fig.add_gridspec(
        rows, 2,
        width_ratios=[3, 2],
        height_ratios=[1] * rows,
        wspace=0.35,
        hspace=0.45,
    )

    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    ax3d.set_title("LASDRA (desired vs actual)", fontsize=18)

    axbars = [fig.add_subplot(gs[r, 1]) for r in range(rows)]

    # ----- world limits -----
    sample = Acum[:min(200, T_all)]
    pos_all = sample[..., :3, 3]  # (T, L+1, 3)
    if args.fix_origin:
        pos_all = pos_all - pos_all[:, :1, :]
    p = pos_all.reshape(-1, 3)
    pmin, pmax = p.min(axis=0), p.max(axis=0)
    span = (pmax - pmin).max()
    center = (pmax + pmin) / 2.0
    lim = span * 0.8 if span > 0 else 0.5
    ax3d.set_xlim(center[0] - lim, center[0] + lim)
    ax3d.set_ylim(center[1] - lim, center[1] + lim)
    ax3d.set_zlim(center[2] - lim, center[2] + lim)

    base_elev = ax3d.elev
    base_azim = ax3d.azim

    # ----- link lines & nodes -----
    desired_lines, actual_lines = [], []
    for _ in range(L):
        d_ln, = ax3d.plot([], [], [], linestyle="--", color="g", lw=2.0, alpha=1.0)
        a_ln, = ax3d.plot([], [], [], linestyle="-", color="k", lw=2.5, alpha=0.45)
        desired_lines.append(d_ln)
        actual_lines.append(a_ln)

    desired_nodes = ax3d.scatter([], [], [], s=15, c="g", alpha=1.0)
    actual_nodes = ax3d.scatter([], [], [], s=18, c="k", alpha=0.6)

    base_marker = ax3d.scatter([0], [0], [0], s=120, marker="o",
                               facecolor="k", edgecolor="y",
                               linewidth=2.0, alpha=1.0, zorder=5)
    base_text = ax3d.text(0.05, 0.05, 0.05, "BASE", color="y",
                          fontsize=10, ha="left", va="bottom")

    # ----- motor visuals (8 per link) -----
    stems_lines = [[None] * mpl for _ in range(L)]
    blade_patches = [[[] for _ in range(mpl)] for _ in range(L)]
    fault_texts = [[None] * mpl for _ in range(L)]

    for li in range(L):
        for mj in range(mpl):
            ln_stem, = ax3d.plot([], [], [], color="k", lw=1.2, alpha=args.stem_alpha)
            stems_lines[li][mj] = ln_stem

            patches = []
            for _ in range(args.prop_blades):
                poly = Poly3DCollection(
                    [np.zeros((4, 3))],
                    closed=True,
                    facecolor="k",
                    edgecolor="none",
                    alpha=args.prop_alpha,
                )
                ax3d.add_collection3d(poly)
                patches.append(poly)
            blade_patches[li][mj] = patches

            txt = ax3d.text(
                0.0,
                0.0,
                0.0,
                "",
                zdir=None,
                transform=ax3d.transData,
                color="r",
                fontsize=8,
                ha="center",
                va="bottom",
            )
            txt.set_visible(False)
            fault_texts[li][mj] = txt

    def link_motor_slice(link_idx: int) -> Tuple[int, int]:
        j0 = link_idx * mpl
        j1 = j0 + mpl
        return j0, j1

    # ----- right-side bar plots -----
    axbars_objs = []
    width = 0.35
    for r, ax in enumerate(axbars):
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Link {r+1} (M1–M{mpl})")
        idxs = np.arange(mpl)
        x_gt = idxs - width / 2.0
        x_pred = idxs + width / 2.0

        bars_gt = ax.bar(
            x_gt,
            np.zeros(mpl),
            width=width,
            alpha=0.35,
            linewidth=1.0,
            edgecolor="gray",
            hatch="//",
            label="REAL VALUE (fault=1)",
        )
        bars_pd = ax.bar(
            x_pred,
            np.zeros(mpl),
            width=width,
            label="Pred prob(fault)",
        )

        ax.set_xticks(idxs)
        ax.set_xticklabels([f"M{i+1}" for i in range(mpl)], rotation=0)
        if r == 0:
            ax.legend(loc="upper right")
        gt_txts = [
            ax.text(i, 1.02, "", ha="center", va="bottom", fontsize=8) for i in idxs
        ]
        axbars_objs.append((bars_gt, bars_pd, gt_txts))

    legend_lines = [
        plt.Line2D([0], [0], color="k", lw=2.5, label="Actual Link", alpha=0.45),
        plt.Line2D([0], [0], color="g", lw=2.0, linestyle="--", label="Desired Link"),
        plt.Line2D([0], [0], marker="$x$", color="k", lw=0, markersize=8, label="Motor"),
        plt.Line2D([0], [0], marker="$x$", color="r", lw=0, markersize=8, label="Faulty Motor (K-of-N)"),
    ]
    ax3d.legend(handles=legend_lines, loc="upper left")

    status_txt = ax3d.text2D(0.58, 0.92, "", transform=ax3d.transAxes, fontsize=14)

    fault_status_txt = fig.text(
        0.12,   
        0.1,    
        "",
        transform=fig.transFigure,
        fontsize=15,
        color="red",
        ha="left",
        va="bottom",
    )

    # ----- playback state -----
    t_idx = [0]  # mutable int
    prob_motor_last = np.zeros(M_total, dtype=float)
    fault_idx_last = 0  # 0=BG, 1..M

    interval_ms = max(
        1, int(1000.0 / max(1e-6, data_hz * args.speed))
    )

    # =======================================================
    #                        Update
    # =======================================================

    def update(_frame):
        nonlocal fault_idx_last
        t = t_idx[0]
        if t >= T_all:
            return []

        prob_motor_last[:] = pred_fault[t]
        fault_idx_last = int(kofn_idx[t])

        t_real = time_axis[t]
        az = base_azim + args.rotate_speed * t_real
        ax3d.view_init(elev=base_elev, azim=az)

        # ---- poses ----
        Td = Dcum[t]
        Ta = Acum[t]
        P_d = Td[:, :3, 3]
        P_a = Ta[:, :3, 3]
        if args.fix_origin:
            P_d = P_d - P_d[:1, :]
            P_a = P_a - P_a[:1, :]

        # links
        for i in range(L):
            xd = [P_d[i, 0], P_d[i + 1, 0]]
            yd = [P_d[i, 1], P_d[i + 1, 1]]
            zd = [P_d[i, 2], P_d[i + 1, 2]]
            desired_lines[i].set_data(xd, yd)
            desired_lines[i].set_3d_properties(zd)

            xa = [P_a[i, 0], P_a[i + 1, 0]]
            ya = [P_a[i, 1], P_a[i + 1, 1]]
            za = [P_a[i, 2], P_a[i + 1, 2]]
            actual_lines[i].set_data(xa, ya)
            actual_lines[i].set_3d_properties(za)

        desired_nodes._offsets3d = (P_d[:, 0], P_d[:, 1], P_d[:, 2])
        actual_nodes._offsets3d = (P_a[:, 0], P_a[:, 1], P_a[:, 2])

        # ---- motor geometry ----
        for i in range(L):
            R_start = Ta[i, :3, :3]
            R_end = Ta[i + 1, :3, :3]

            ratio = float(args.anchor_ratio)
            p_front = (1.0 - ratio) * P_a[i] + ratio * P_a[i + 1]
            p_back = ratio * P_a[i] + (1.0 - ratio) * P_a[i + 1]

            y_end, z_end = R_end[:, 1], R_end[:, 2]
            y_sta, z_sta = R_start[:, 1], R_start[:, 2]

            four_front = np.array(
                [
                    p_front + args.arm_len * y_end,
                    p_front - args.arm_len * y_end,
                    p_front + args.arm_len * z_end,
                    p_front - args.arm_len * z_end,
                ]
            )
            four_back = np.array(
                [
                    p_back + args.arm_len * y_sta,
                    p_back - args.arm_len * y_sta,
                    p_back + args.arm_len * z_sta,
                    p_back - args.arm_len * z_sta,
                ]
            )
            motor_pos   = np.vstack([four_back, four_front])  # (8,3)
            R_for_blade = [R_start] * 4 + [R_end] * 4
            anchors     = [p_back] * 4 + [p_front] * 4

            for j in range(mpl):
                cls1_based = i * mpl + j + 1  # 1..M
                is_fault = (fault_idx_last == cls1_based)
                color_face = "r" if is_fault else "k"

                pj = motor_pos[j]
                p_anc = anchors[j]
                R_ref = R_for_blade[j]

                # stem
                stems_lines[i][j].set_data([p_anc[0], pj[0]], [p_anc[1], pj[1]])
                stems_lines[i][j].set_3d_properties([p_anc[2], pj[2]])
                stems_lines[i][j].set_color(color_face)

                # blades
                n_hat = _norm(pj - p_anc)
                u_ref = R_ref[:, 1]
                v_ref = R_ref[:, 2]
                u = u_ref - np.dot(u_ref, n_hat) * n_hat
                if np.linalg.norm(u) < 1e-6:
                    u = v_ref - np.dot(v_ref, n_hat) * n_hat
                u = u / (np.linalg.norm(u) + 1e-9)
                v = np.cross(n_hat, u)
                v = v / (np.linalg.norm(v) + 1e-9)

                spin_sign = 1.0 if (args.spin_dir_alt == 0 or (j % 2 == 0)) else -1.0
                base_phase = (
                    0.0
                    if is_fault
                    else spin_sign * 2.0 * math.pi * args.prop_rps * (1.0 / data_hz) * t
                )

                for k, poly in enumerate(blade_patches[i][j]):
                    theta = base_phase + 2.0 * math.pi * (k / max(1, args.prop_blades))
                    c = math.cos(theta)
                    s = math.sin(theta)
                    axis = (c * u + s * v)
                    perp = (-s * u + c * v)

                    r_root = 0.25 * args.prop_radius
                    r_tip = args.prop_radius
                    half_c = 0.5 * args.prop_chord

                    root = pj + r_root * axis
                    tip = pj + r_tip * axis
                    p1 = root + half_c * perp
                    p2 = tip + half_c * perp
                    p3 = tip - half_c * perp
                    p4 = root - half_c * perp
                    quad = np.stack([p1, p2, p3, p4], axis=0)

                    poly.set_verts([quad])
                    poly.set_facecolor(color_face)
                    poly.set_edgecolor("none")
                    poly.set_alpha(args.prop_alpha)

                label = fault_texts[i][j]
                if is_fault:
                    offset = 0.06  
                    x3, y3, z3 = pj + offset * np.array([0.0, 0.0, 1.0])
                    x2, y2, _ = proj3d.proj_transform(x3, y3, z3, ax3d.get_proj())
                    prob = float(prob_motor_last[cls1_based - 1])
                    label.set_text(f"L{i+1} M{j+1} Fault\np={prob:.2f}")
                    label.set_position((x2, y2))
                    label.set_visible(True)
                else:
                    label.set_visible(False)

        # ---- right-side bars ----
        for r in range(rows):
            j0, j1 = link_motor_slice(r)
            bars_gt, bars_pred, gt_txts = axbars_objs[r]
            gt_slice = gt_fault[t, j0:j1]
            pred_slice = pred_fault[t, j0:j1]

            for i_m in range(mpl):
                m_idx = j0 + i_m
                bars_gt[i_m].set_height(float(gt_slice[i_m]))
                val = float(pred_slice[i_m])
                if not np.isfinite(val):
                    val = 0.0
                bars_pred[i_m].set_height(val)

                is_alarm = (fault_idx_last == (m_idx + 1))
                bars_pred[i_m].set_edgecolor("r" if is_alarm else "black")
                bars_pred[i_m].set_linewidth(2.5 if is_alarm else 0.5)

                gt_txts[i_m].set_text("GT:F" if gt_slice[i_m] >= 0.5 else "")

        status_txt.set_text(f"t = {t_real:4.2f}s")

        if (t_fault_pred is not None) and (t >= t_fault_pred):
            if (link_pred is not None):
                msg1 = f"FAULT DETECTED!!  Pred: L{link_pred} M{motor_pred}"
                if link_gt is not None:
                    msg1 += f"  |  GT: L{link_gt} M{motor_gt}"
                if delay_s is not None:
                    msg2 = f"Latency: {delay_s:.3f}s"
                else:
                    msg2 = ""
                fault_status_txt.set_text(msg1 + "\n" + msg2)
                fault_status_txt.set_visible(True)
            else:
                fault_status_txt.set_visible(False)
        else:
            fault_status_txt.set_visible(False)

        t_idx[0] += 1

        artists = (
            desired_lines
            + actual_lines
            + [desired_nodes, actual_nodes, base_marker, base_text, status_txt, fault_status_txt]
        )
        for i in range(L):
            artists.extend(stems_lines[i])
            for patches in blade_patches[i]:
                artists.extend(patches)
            artists.extend(fault_texts[i])
        for r in range(rows):
            bars_gt, bars_pred, gt_txts = axbars_objs[r]
            artists.extend(list(bars_gt))
            artists.extend(list(bars_pred))
            artists.extend(gt_txts)
        return artists

    # =======================================================
    #                   Animation / Save / Show
    # =======================================================
    ani = FuncAnimation(
        fig,
        update,
        frames=T_all,
        interval=interval_ms,
        blit=False,
        save_count=T_all,
        cache_frame_data=False,
    )
    plt.tight_layout()

    def _parse_bitrate_to_kbps(b):
        s = str(b).strip().lower()
        if s.endswith("k"):
            s = s[:-1]
        return int(float(s))

    if args.save_video:
        ext = os.path.splitext(args.out)[1].lower()
        try:
            if ext in [".mp4", ".m4v", ".mov"]:
                writer = FFMpegWriter(
                    fps=args.video_fps,
                    codec=args.codec,
                    bitrate=_parse_bitrate_to_kbps(args.bitrate),
                )
            elif ext in [".gif"]:
                writer = PillowWriter(fps=args.video_fps)
            else:
                raise ValueError(f"Unsupported extension: {ext} (use .mp4 or .gif)")

            print(f"[INFO] Saving video to: {args.out}  (fps={args.video_fps}, dpi={args.dpi})")
            ani.save(args.out, writer=writer, dpi=args.dpi)
            print("[INFO] Done.")
        except Exception as e:
            print(f"[ERROR] Video save failed: {e}")
        finally:
            plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
"""
python3 visualize_lasdra.py \
  --ckpt /home/user/transformer_fault_diagnosis/GraphFDI_Transformer_L5/GraphFDI_Transformer_L5.pth \
  --data_dir /home/user/transformer_fault_diagnosis/data_storage/link_5 \
  --znorm /home/user/transformer_fault_diagnosis/GraphFDI_Transformer_L5/znorm_link5.npz \
  --shard_first 1 --shard_last 5 \
  --seq_idx 12 \
  --motor_geom /home/user/transformer_fault_diagnosis/motor_geom_link5.npz \
  --theta_motor 0.8 \
  --kofn_k 8 \
  --kofn_n 10 \
  --save_video 0
"""


"""
python3 visualize_lasdra.py \
  --ckpt /home/user/transformer_fault_diagnosis/GraphFDI_Transformer_L5/GraphFDI_Transformer_L5.pth \
  --data_dir /home/user/transformer_fault_diagnosis/data_storage/link_5 \
  --znorm /home/user/transformer_fault_diagnosis/GraphFDI_Transformer_L5/znorm_link5.npz \
  --shard_first 1 --shard_last 5 \
  --seq_idx 12 \
  --motor_geom /home/user/transformer_fault_diagnosis/motor_geom_link5.npz \
  --theta_motor 0.8 \
  --kofn_k 8 \
  --kofn_n 10 \
  --save_video 1 \
  --out /home/user/transformer_fault_diagnosis/lasdra_link5.mp4 \
  --video_fps 60 \
  --dpi 150
"""