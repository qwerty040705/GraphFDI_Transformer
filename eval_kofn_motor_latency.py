#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from typing import Optional, Dict, Tuple, List, Set

import numpy as np
import torch
from tqdm import tqdm

from graph_fdi_train import (
    LASDRAFDIDataset,
    MotorGeom,
    GraphFDIModel,
    setup_device,
)


def kofn_latched_from_probs(
    probs: np.ndarray,
    K: int = 3,
    N: int = 5,
    thresh: float = 0.8,
) -> np.ndarray:
    T = len(probs)
    buf = np.zeros(N, dtype=np.float32)
    out = np.zeros(T, dtype=np.int32)
    latched = 0
    for t in range(T):
        if latched == 0:
            buf[t % N] = 1.0 if probs[t] >= thresh else 0.0
            if buf.sum() >= K:
                latched = 1
        out[t] = latched
    return out


def build_model_from_ckpt(
    ckpt_path: str,
    device: torch.device,
    links: int,
    motors_per_link: int = 8,
) -> GraphFDIModel:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg_dict: Dict = ckpt.get("cfg", {})

    d_model = cfg_dict.get("d_model", 384)
    heads = cfg_dict.get("heads", 6)
    depth = cfg_dict.get("depth", 5)
    dropout = cfg_dict.get("dropout", 0.1)
    temporal = cfg_dict.get("temporal", "transformer")

    d_node_in = 10
    d_edge_in = 16

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
    p = argparse.ArgumentParser(
        description="Eval GraphFDI with motor-wise K-of-N & latency on TEST split"
    )
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--shard_first", type=int, default=1)
    p.add_argument("--shard_last", type=int, default=5)
    p.add_argument("--motor_geom", type=str, default=None)
    p.add_argument("--device", type=str, default="auto")

    p.add_argument("--T_win", type=int, default=128)
    p.add_argument("--stride", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--K", type=int, default=3)
    p.add_argument("--N", type=int, default=5)
    p.add_argument("--thresh", type=float, default=0.8)

    p.add_argument("--cache_windows", type=int, default=1)
    p.add_argument("--cache_dir", type=str, default="./cache_eval")
    p.add_argument("--cache_dtype", type=str, default="fp32",
                   choices=["fp32", "fp16"])

    p.add_argument("--max_test_seqs", type=int, default=0)
    p.add_argument("--max_T_steps", type=int, default=0)

    args = p.parse_args()

    device = setup_device(args.device)

    shard_ids = list(range(args.shard_first, args.shard_last + 1))
    print(f"[INFO] Using shards: {shard_ids}")

    geom = MotorGeom(args.motor_geom) if args.motor_geom else None

    full_ds = LASDRAFDIDataset(
        data_dir=args.data_dir,
        shard_ids=shard_ids,
        T_win=args.T_win,
        stride=args.stride,
        motors_per_link=8,
        normalize=True,
        motor_geom=geom,
        cache_windows=bool(args.cache_windows),
        cache_dir=args.cache_dir,
        cache_dtype=args.cache_dtype,
    )

    L = full_ds.L
    mpl = full_ds.mpl
    M_total = L * mpl
    print(f"[INFO] link_count = {L}, motors_per_link = {mpl} -> total motors = {M_total}")

    n = len(full_ds)
    idx = np.arange(n)
    np.random.seed(args.seed)
    np.random.shuffle(idx)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    test_indices = idx[n_train + n_val :].tolist()

    print(f"[INFO] n={n}, train={n_train}, val={n_val}, test={len(test_indices)}")

    test_seq_set: Set[Tuple[int, int]] = set()
    for ds_idx in test_indices:
        sh_idx, s, t0 = full_ds.index[ds_idx]
        test_seq_set.add((sh_idx, s))

    test_seq_list = sorted(test_seq_set)
    if args.max_test_seqs > 0:
        test_seq_list = test_seq_list[: args.max_test_seqs]

    print(f"[INFO] Unique TEST sequences (after limit) = {len(test_seq_list)}")

    model = build_model_from_ckpt(args.ckpt, device, links=L, motors_per_link=mpl)

    tp = fp = tn = fn = 0
    latencies: List[float] = []

    T_win = full_ds.T_win
    mu = full_ds._mu
    std = full_ds._std

    model.eval()
    with torch.no_grad():
        for idx_seq, (sh_idx, s) in enumerate(
            tqdm(test_seq_list, desc="Evaluating sequences", ncols=100), start=1
        ):
            pack = full_ds.samples[sh_idx]
            label_full = pack["label"][s]
            T_all = label_full.shape[0]
            if args.max_T_steps > 0:
                T_all = min(T_all, args.max_T_steps)
            dt = float(pack.get("dt", 1.0))

            if label_full.shape[1] != M_total:
                raise RuntimeError(
                    f"label dim mismatch: got {label_full.shape[1]}, expected {M_total}"
                )

            gt_fault = (label_full[:T_all] == 0).astype(np.int32)
            pred_prob = np.full((T_all, M_total), 0.0, dtype=np.float32)

            for t_cur in range(T_all):
                if t_cur < T_win - 1:
                    continue

                t0 = t_cur - T_win + 1

                node_feat, edge_feat, attn_mask, _ = full_ds._extract_window(
                    sh_idx, s, t0
                )

                if (mu is not None) and (std is not None):
                    L_links = full_ds.L
                    node_feat[:, :L_links, 0:9] = (
                        (node_feat[:, :L_links, 0:9] - mu) / std
                    )

                node_feat = np.nan_to_num(node_feat, nan=0.0, posinf=1e6, neginf=-1e6)
                edge_feat = np.nan_to_num(edge_feat, nan=0.0, posinf=1e6, neginf=-1e6)

                X = torch.from_numpy(node_feat).float().unsqueeze(0).to(device)
                E = torch.from_numpy(edge_feat).float().unsqueeze(0).to(device)
                M_mask = torch.from_numpy(attn_mask).unsqueeze(0).unsqueeze(0).to(device)

                o1, o2, m_logits = model(X, E, M_mask)

                p_onset = torch.sigmoid(o1[0]).item()
                link_probs = torch.softmax(o2[0], dim=-1)

                motor_probs_links = []
                for li in range(L):
                    probs_li = torch.softmax(m_logits[li][0], dim=-1)
                    motor_probs_links.append(probs_li.cpu().numpy())

                for li in range(L):
                    p_link_li = float(link_probs[li + 1].item())
                    for mj in range(mpl):
                        p_mj_given_li = float(motor_probs_links[li][mj])
                        m_idx = li * mpl + mj
                        pred_prob[t_cur, m_idx] = (
                            p_onset * p_link_li * p_mj_given_li
                        )

            start_t = T_win - 1

            for m in range(M_total):
                gt_seq = gt_fault[start_t:, m]
                probs_seq = pred_prob[start_t:, m]

                pred_seq = kofn_latched_from_probs(
                    probs_seq, K=args.K, N=args.N, thresh=args.thresh
                )

                tp += int(np.sum((pred_seq == 1) & (gt_seq == 1)))
                tn += int(np.sum((pred_seq == 0) & (gt_seq == 0)))
                fp += int(np.sum((pred_seq == 1) & (gt_seq == 0)))
                fn += int(np.sum((pred_seq == 0) & (gt_seq == 1)))

                if np.any(gt_seq == 1):
                    t_onset_rel = int(np.argmax(gt_seq == 1))
                    detect_mask = (pred_seq == 1)
                    idx_detect = np.where(detect_mask & (np.arange(len(pred_seq)) >= t_onset_rel))[0]
                    if idx_detect.size > 0:
                        t_detect_rel = int(idx_detect[0])
                        t_onset_abs = start_t + t_onset_rel
                        t_detect_abs = start_t + t_detect_rel
                        latency_sec = (t_detect_abs - t_onset_abs) * dt
                        latencies.append(latency_sec)

    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    print("\n================= TEST Metrics (K-of-N on motor probs) =================")
    print(f"TP={tp}, FP={fp}, TN={tn}, FN={fn}, total={total}")
    print(f"Accuracy = {acc:.4f}")
    print(f"Precision = {prec:.4f}")
    print(f"Recall    = {rec:.4f}")
    print(f"F1        = {f1:.4f}")

    if len(latencies) > 0:
        lat_arr = np.array(latencies, dtype=np.float64)
        q1, q2, q3 = np.percentile(lat_arr, [25, 50, 75])
        print("\nLatency (only for detected faulty motors)")
        print(f"Count = {len(lat_arr)}")
        print(f"Q1    = {q1:.4f} s")
        print(f"Q2    = {q2:.4f} s (median)")
        print(f"Q3    = {q3:.4f} s")
    else:
        print("\n[WARN] No detected faults => latency stats not available.")


if __name__ == "__main__":
    main()


"""
python3 eval_kofn_motor_latency.py \
  --ckpt /home/user/transformer_fault_diagnosis/GraphFDI_Transformer_L1/GraphFDI_Transformer_L1.pth \
  --data_dir /home/user/transformer_fault_diagnosis/data_storage/link_1 \
  --motor_geom /home/user/transformer_fault_diagnosis/motor_geom_link1.npz \
  --shard_first 1 --shard_last 3 \
  --T_win 128 --stride 64 \
  --K 3 --N 5 --thresh 0.8 \
  --seed 42 \
  --device cuda \
  --cache_windows 1 \
  --cache_dir /home/user/transformer_fault_diagnosis/cache_eval_L1 \
  --cache_dtype fp32 \
  --max_test_seqs 100
  

python3 eval_kofn_motor_latency.py \
  --ckpt /home/user/transformer_fault_diagnosis/GraphFDI_Transformer_L2/GraphFDI_Transformer_L2.pth \
  --data_dir /home/user/transformer_fault_diagnosis/data_storage/link_2 \
  --motor_geom /home/user/transformer_fault_diagnosis/motor_geom_link2.npz \
  --shard_first 1 --shard_last 4 \
  --T_win 128 --stride 64 \
  --K 3 --N 5 --thresh 0.8 \
  --seed 42 \
  --device cuda \
  --cache_windows 1 \
  --cache_dir /home/user/transformer_fault_diagnosis/cache_eval_L2 \
  --cache_dtype fp32 \
  --max_test_seqs 100

python3 eval_kofn_motor_latency.py \
  --ckpt /home/user/transformer_fault_diagnosis/GraphFDI_Transformer_L3/GraphFDI_Transformer_L3.pth \
  --data_dir /home/user/transformer_fault_diagnosis/data_storage/link_3 \
  --motor_geom /home/user/transformer_fault_diagnosis/motor_geom_link3.npz \
  --shard_first 1 --shard_last 5 \
  --T_win 128 --stride 64 \
  --K 3 --N 5 --thresh 0.8 \
  --seed 42 \
  --device cuda \
  --cache_windows 1 \
  --cache_dir /home/user/transformer_fault_diagnosis/cache_eval_L3 \
  --cache_dtype fp32 \
  --max_test_seqs 100

python3 eval_kofn_motor_latency.py \
  --ckpt /home/user/transformer_fault_diagnosis/GraphFDI_Transformer_L4/GraphFDI_Transformer_L4.pth \
  --data_dir /home/user/transformer_fault_diagnosis/data_storage/link_4 \
  --motor_geom /home/user/transformer_fault_diagnosis/motor_geom_link4.npz \
  --shard_first 1 --shard_last 5 \
  --T_win 128 --stride 64 \
  --K 3 --N 5 --thresh 0.8 \
  --seed 42 \
  --device cuda \
  --cache_windows 1 \
  --cache_dir /home/user/transformer_fault_diagnosis/cache_eval_L4 \
  --cache_dtype fp32 \
  --max_test_seqs 100

python3 eval_kofn_motor_latency.py \
  --ckpt /home/user/transformer_fault_diagnosis/GraphFDI_Transformer_L5/GraphFDI_Transformer_L5.pth \
  --data_dir /home/user/transformer_fault_diagnosis/data_storage/link_5 \
  --motor_geom /home/user/transformer_fault_diagnosis/motor_geom_link5.npz \
  --shard_first 1 --shard_last 5 \
  --T_win 128 --stride 64 \
  --K 3 --N 5 --thresh 0.8 \
  --seed 42 \
  --device cuda \
  --cache_windows 1 \
  --cache_dir /home/user/transformer_fault_diagnosis/cache_eval_L5 \
  --cache_dtype fp32 \
  --max_test_seqs 100
"""