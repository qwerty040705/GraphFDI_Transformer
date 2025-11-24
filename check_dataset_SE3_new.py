# check_dataset_SE3_new.py
import os
import glob
import numpy as np

np.set_printoptions(precision=6, suppress=True)

# ---------- helpers ----------
def decompose_se3(T4):
    R = T4[:3, :3]
    p = T4[:3, 3]
    return R, p

def rot_angle(R_err):
    tr = float(np.trace(R_err))
    c = max(-1.0, min(1.0, (tr - 1.0) / 2.0))  # numerical safety
    return float(np.arccos(c))

def load_single_or_sharded(link_count, target_sample_idx=None):
    """
    - 단일 파일(data_storage/link_{N}/fault_dataset.npz)이 있으면 그걸 로드
    - 없으면 샤드(data_storage/link_{N}/fault_dataset_shard_*.npz)를 탐색해서
      - target_sample_idx가 주어졌으면 해당 샤드만 로드 (메모리 절약)
      - 안 주어졌으면 모든 샤드를 합쳐서 로드 (주의: 메모리 사용량 큼)
    반환:
      data(dict-like np.load), info(dict: mode, shard_paths, shard_index, local_idx, total_samples)
    """
    data_dir = os.path.join("data_storage", f"link_{link_count}")
    single_path = os.path.join(data_dir, "fault_dataset.npz")
    shard_paths = sorted(glob.glob(os.path.join(data_dir, "fault_dataset_shard_*.npz")))

    info = {
        "mode": None,              # "single" | "sharded-one" | "sharded-all"
        "data_dir": data_dir,
        "single_path": single_path,
        "shard_paths": shard_paths,
        "picked_shard_path": None,
        "picked_shard_index": None,
        "picked_local_index": None,
        "total_samples": None,
        "cum_counts": None,
    }

    if os.path.exists(single_path):
        data = np.load(single_path, allow_pickle=True)
        S = data['desired_link_cum'].shape[0]
        info.update({"mode": "single", "total_samples": S})
        return data, info

    if not shard_paths:
        raise FileNotFoundError(
            f"No dataset found.\n"
            f" - Tried single: {single_path}\n"
            f" - Tried shards: {os.path.join(data_dir, 'fault_dataset_shard_*.npz')}"
        )

    # 샤드 개수/샘플 수 스캔
    counts = []
    for p in shard_paths:
        with np.load(p, allow_pickle=True) as z:
            counts.append(int(z['desired_link_cum'].shape[0]))
    cum = np.cumsum([0] + counts)  # 길이 = len(shard_paths)+1
    S_total = cum[-1]
    info.update({"total_samples": S_total, "cum_counts": cum})

    if target_sample_idx is None:
        # 모든 샤드를 합쳐서 메모리에 로드 (권장: 특정 샘플만 볼 땐 target_sample_idx 지정)
        # necessary keys만 합치자
        keys_float = [
            "desired_link_rel", "actual_link_rel",
            "desired_link_cum", "actual_link_cum",
            "desired_ee", "actual_ee",
        ]
        keys_int = ["label", "onset_idx", "first_fault_t", "t0", "dof", "joint_counts"]
        # 일부 키(예: desired_ee/actual_ee)는 없을 수도 있음 → 존재하는 키만 수집
        acc = {}
        present_keys = set()

        # 첫 샤드의 메타 키 복사 (timestamps, dt, link_count)
        with np.load(shard_paths[0], allow_pickle=True) as z0:
            base_meta = {
                "timestamps": z0["timestamps"],
                "dt": float(z0["dt"]),
                "link_count": int(z0["link_count"]),
            }

        def _append_key(k, arr):
            if k not in acc:
                acc[k] = [arr]
                present_keys.add(k)
            else:
                acc[k].append(arr)

        for p in shard_paths:
            z = np.load(p, allow_pickle=True)
            # floats
            for k in keys_float:
                if k in z.files:
                    _append_key(k, z[k])
            # ints/labels
            for k in keys_int:
                if k in z.files:
                    _append_key(k, z[k])
            z.close()

        # 스택
        out = {}
        for k in present_keys:
            # label/object 처리: 아래 본 로직에서 다시 안전 변환하므로 일단 그냥 붙임
            out[k] = np.concatenate(acc[k], axis=0)

        # 메타 다시 붙이기
        out.update(base_meta)
        info["mode"] = "sharded-all"
        return out, info

    # target_sample_idx가 지정됨 → 그 샤드만 로드
    sidx = int(target_sample_idx)
    if not (0 <= sidx < S_total):
        raise IndexError(f"sample_idx out of range: {sidx} (total_samples={S_total})")
    # 샤드 찾기: cum[i] <= sidx < cum[i+1]
    shard_i = None
    for i in range(len(shard_paths)):
        if cum[i] <= sidx < cum[i+1]:
            shard_i = i
            local = sidx - cum[i]
            break
    assert shard_i is not None
    picked = shard_paths[shard_i]
    info.update({
        "mode": "sharded-one",
        "picked_shard_path": picked,
        "picked_shard_index": shard_i,
        "picked_local_index": int(local),
    })
    data = np.load(picked, allow_pickle=True)
    return data, info

# ---------- I/O ----------
link_count = int(input("How many links do you want to check?: ").strip())
data_dir = os.path.join("data_storage", f"link_{link_count}")

# 확인하고 싶은 샘플 인덱스 (전역 인덱스; 샤드 전체를 0..S_total-1로 본 번호)
sample_idx = 1 

# 데이터 로드 (가능하면 해당 샤드 1개만 로드)
data, info = load_single_or_sharded(link_count, target_sample_idx=sample_idx)

print("=== DATA MODE ===")
print(" mode        :", info["mode"])
print(" data_dir    :", info["data_dir"])
if info["mode"] == "single":
    print(" file        :", info["single_path"])
elif info["mode"] == "sharded-one":
    print(" shard file  :", info["picked_shard_path"])
    print(" shard index :", info["picked_shard_index"])
    print(" local index :", info["picked_local_index"])
    print(" total S     :", info["total_samples"])
elif info["mode"] == "sharded-all":
    print(" shards      :", len(info["shard_paths"]))
    print(" total S     :", info["total_samples"])
print()

# ====== Keys ======
has_rel = ('desired_link_rel' in data.files) and ('actual_link_rel' in data.files)
has_cum = ('desired_link_cum' in data.files) and ('actual_link_cum' in data.files)
if not has_cum:
    raise KeyError("Dataset must include desired_link_cum / actual_link_cum to reconstruct EE.")

desired_link_rel = data['desired_link_rel'] if has_rel else None
actual_link_rel  = data['actual_link_rel']  if has_rel else None
desired_link_cum = data['desired_link_cum']           # (S,T,N,4,4)
actual_link_cum  = data['actual_link_cum']            # (S,T,N,4,4)

# ✅ label이 object 배열이면 numeric으로 스택 (S,T,8*link_count)
label = data['label']
if label.dtype == object:
    label = np.stack([np.asarray(x) for x in label], axis=0)
label = label.astype(np.int32, copy=False)

# meta
if 'dof' in data.files and np.ndim(data['dof']) == 0:
    dof = int(data['dof'])
else:
    # 샤드 합치기 모드 등에서 dof가 배열일 수 있음 → 첫 값 사용
    dof = int(np.asarray(data['dof']).ravel()[0]) if 'dof' in data.files else link_count

# ---------- Shapes / format ----------
# EE = 마지막 누적변환 (T_{0->N})
desired_ee = desired_link_cum[:, :, -1]   # (S,T,4,4)
actual_ee  = actual_link_cum[:, :, -1]    # (S,T,4,4)

S, T, _, _ = desired_ee.shape
motor_count = label.shape[2]

fmt_parts = ["EE(derived from T_{0->N})"]
if has_rel: fmt_parts.append("PER_LINK_REL")
if has_cum: fmt_parts.append("PER_LINK_CUM")
fmt = " + ".join(fmt_parts)

print(f"Format: {fmt}")
print(f"desired_ee.shape   = {desired_ee.shape}")
print(f"actual_ee.shape    = {actual_ee.shape}")
if has_rel:
    print(f"desired_link_rel   = {desired_link_rel.shape}  # T_(i->i+1)")
    print(f"actual_link_rel    = {actual_link_rel.shape}")
print(f"desired_link_cum   = {desired_link_cum.shape}  # T_(0->k+1)")
print(f"actual_link_cum    = {actual_link_cum.shape}")
print(f"label.shape        = {label.shape} (motor_count={motor_count})")
print(f"link_count (N)     = {link_count}")
print(f"dof (total joints) = {dof}")

# ---------- Time indices ----------
raw_times = [0, 1, 10, 50, 100, 150, 200, 250, 500, 750, 800, 999]
time_idx_list = [t for t in raw_times if t < T]

# ---------- Resolve sample index (local for sharded-one) ----------
if info["mode"] == "sharded-one":
    s_local = info["picked_local_index"]
else:
    s_local = sample_idx
assert 0 <= s_local < S, f"local sample index out of range: {s_local} (S={S})"

# ---------- Inspection ----------
for time_idx in time_idx_list:
    print("\n" + "="*50)
    # 전역 인덱스도 같이 보여주기
    if info["mode"] == "sharded-one":
        global_idx = sample_idx
        print(f"[Sample {global_idx} (local {s_local}), Time {time_idx}]")
    else:
        print(f"[Sample {s_local}, Time {time_idx}]")

    lbl = label[s_local, time_idx].astype(int)
    faulty_idx = np.where(lbl == 0)[0]
    print("Motor labels:", lbl)
    if faulty_idx.size > 0:
        print(f"→ Faulty motors at this time: {faulty_idx.tolist()}")

    # --- EE = 마지막 누적변환 ---
    T_des_ee = desired_ee[s_local, time_idx]
    T_act_ee = actual_ee[s_local, time_idx]
    R_des_ee, p_des_ee = decompose_se3(T_des_ee)
    R_act_ee, p_act_ee = decompose_se3(T_act_ee)

    print("\n[EE=T_{0->N}] Desired:")
    print(T_des_ee)
    print("\n[EE=T_{0->N}] Actual :")
    print(T_act_ee)

    ee_pos_err = np.linalg.norm(p_act_ee - p_des_ee)
    R_err_ee   = R_des_ee.T @ R_act_ee
    ee_ang_err = rot_angle(R_err_ee)
    ee_ang_deg = np.degrees(ee_ang_err)

    print(f"\n[EE=T0N] Position error = {ee_pos_err:.6f} m")
    print(f"[EE=T0N] Orientation error = {ee_ang_err:.6f} rad ({ee_ang_deg:.3f} deg)")

    # --- Per-link relative transforms ---
    if has_rel:
        per_link_pos = []
        per_link_ang = []

        print("\nPer-link relative transforms & errors (T_{i-1,i}):")
        for i in range(link_count):
            T_rel_des = desired_link_rel[s_local, time_idx, i]
            T_rel_act = actual_link_rel[s_local, time_idx, i]
            R_rel_des, p_rel_des = decompose_se3(T_rel_des)
            R_rel_act, p_rel_act = decompose_se3(T_rel_act)

            print(f"\n  - T_{{{i}->{i+1}}} (Desired):")
            print(T_rel_des)
            print(f"  - T_{{{i}->{i+1}}} (Actual) :")
            print(T_rel_act)

            j_pos = np.linalg.norm(p_rel_act - p_rel_des)
            Rj_err = R_rel_des.T @ R_rel_act
            j_ang = rot_angle(Rj_err)
            j_deg = np.degrees(j_ang)

            per_link_pos.append(j_pos)
            per_link_ang.append(j_ang)

            print(f"    -> pos_err={j_pos:.6f} m, ori_err={j_ang:.6f} rad ({j_deg:.3f} deg)")

        per_link_pos = np.array(per_link_pos)
        per_link_ang = np.array(per_link_ang)
        j_max_pos = int(np.argmax(per_link_pos))
        j_max_ang = int(np.argmax(per_link_ang))

        print("\nSummary at this time (per-link relative):")
        print(f"  mean pos_err = {per_link_pos.mean():.6f} m | "
              f"max={per_link_pos.max():.6f} m @ joint {j_max_pos+1}")
        print(f"  mean ori_err = {per_link_ang.mean():.6f} rad "
              f"({np.degrees(per_link_ang).mean():.3f} deg) | "
              f"max={per_link_ang.max():.6f} rad "
              f"({np.degrees(per_link_ang).max():.3f} deg) @ joint {j_max_ang+1}")
