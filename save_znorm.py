import os
import argparse

import numpy as np
import torch

from graph_fdi_train import (
    LASDRAFDIDataset,
    MotorGeom,
)


def main():
    parser = argparse.ArgumentParser(
        description="Compute & save global Z-Norm (mu, std) for GraphFDI LASDRAFDIDataset."
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to GraphFDI checkpoint (.pth)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to link_3 directory containing fault_dataset_shard_*.npz",
    )
    parser.add_argument(
        "--shard_first",
        type=int,
        default=1,
        help="First shard id (inclusive)",
    )
    parser.add_argument(
        "--shard_last",
        type=int,
        default=5,
        help="Last shard id (inclusive)",
    )
    parser.add_argument(
        "--motor_geom",
        type=str,
        default=None,
        help="Path to motor_geom.npz (same as training)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional: explicit output path for znorm npz. "
             "Default: <ckpt_dir>/znorm_link{L}.npz",
    )

    args = parser.parse_args()

    print(f"[INFO] Loading checkpoint: {args.ckpt}")
    ckpt_tmp = torch.load(args.ckpt, map_location="cpu")
    cfg_ckpt = ckpt_tmp.get("cfg", {})

    T_win = int(cfg_ckpt.get("T_win", 128))
    stride = int(cfg_ckpt.get("stride", 64))
    cache_windows = int(cfg_ckpt.get("cache_windows", 1))
    cache_dtype = str(cfg_ckpt.get("cache_dtype", "fp32"))
    cache_dir = cfg_ckpt.get("cache_dir", "./cache_L3")

    shard_ids = list(range(args.shard_first, args.shard_last + 1))
    print(f"[INFO] Using shards: {shard_ids}")
    print(f"[INFO] T_win={T_win}, stride={stride}, cache_windows={cache_windows}, cache_dtype={cache_dtype}")

    geom = MotorGeom(args.motor_geom) if args.motor_geom else None

    print("[INFO] Building LASDRAFDIDataset and computing global mu/std ...")
    ds = LASDRAFDIDataset(
        data_dir=args.data_dir,
        shard_ids=shard_ids,
        T_win=T_win,
        stride=stride,
        motors_per_link=8,   
        normalize=True,
        motor_geom=geom,
        cache_windows=bool(cache_windows),
        cache_dir=cache_dir,
        cache_dtype=cache_dtype,
    )

    L = ds.L
    mu = getattr(ds, "_mu", None)
    std = getattr(ds, "_std", None)

    if mu is None or std is None:
        raise RuntimeError(
            "Dataset did not compute mu/std. "
            "확인: LASDRAFDIDataset(normalize=True) 로 생성했는지, "
            "그리고 구현에서 _mu, _std를 설정하는지 확인 필요."
        )

    mu = np.asarray(mu, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)

    print(f"[INFO] link_count = {L}")
    print(f"[INFO] mu shape = {mu.shape}, std shape = {std.shape}")
    print(f"[INFO] mu = {mu}")
    print(f"[INFO] std = {std}")

    ckpt_dir = os.path.dirname(os.path.abspath(args.ckpt))
    if args.out is not None:
        out_path = args.out
    else:
        out_path = os.path.join(ckpt_dir, f"znorm_link{L}.npz")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    np.savez(out_path, mu=mu, std=std)
    print(f"[INFO] Saved Z-Norm parameters to: {out_path}")


if __name__ == "__main__":
    main()



"""
python3 compute_znorm.py \
  --ckpt /home/user/transformer_fault_diagnosis/GraphFDI_Transformer_L2/GraphFDI_Transformer_L2.pth \
  --data_dir /home/user/transformer_fault_diagnosis/data_storage/link_2 \
  --shard_first 1 \
  --shard_last 4 \
  --motor_geom /home/user/transformer_fault_diagnosis/motor_geom_link2.npz

"""