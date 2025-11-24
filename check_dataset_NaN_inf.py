# check_dataset_integrity.py
import os
import numpy as np

def fmt(x):
    try:
        return f"{float(x):.3e}"
    except Exception:
        return str(x)

def is_numeric_array(a):
    return isinstance(a, np.ndarray) and np.issubdtype(a.dtype, np.number)

def check_nonfinite(name, arr):
    nan_count = int(np.isnan(arr).sum()) if is_numeric_array(arr) else 0
    inf_count = int(np.isinf(arr).sum()) if is_numeric_array(arr) else 0
    print(f"  • {name:20s} | NaN={nan_count:6d} | Inf={inf_count:6d}", end="")
    if is_numeric_array(arr):
        try:
            amin, amax = np.nanmin(arr), np.nanmax(arr)
            print(f" | min={fmt(amin)} | max={fmt(amax)}")
        except Exception:
            print()
    else:
        print()
    return nan_count, inf_count

def check_se3_block(name, Tset):
    """
    Tset: (..., 4, 4)
    검사: 마지막 행 [0,0,0,1], det(R)≈+1, R^T R≈I
    """
    if Tset.size == 0:
        print(f"  • {name}: empty, skip")
        return

    assert Tset.shape[-2:] == (4, 4), f"{name} must be (...,4,4), got {Tset.shape}"
    Ts = Tset.reshape(-1, 4, 4)

    bottom = Ts[:, 3, :]
    last_row_ok = np.allclose(bottom, np.array([0, 0, 0, 1])[None, :], atol=1e-6)
    bad_last = np.where(~np.isclose(bottom, np.array([0,0,0,1])[None,:], atol=1e-6).all(axis=1))[0]

    R = Ts[:, :3, :3]
    detR = np.linalg.det(R)
    det_ok = np.allclose(detR, 1.0, atol=1e-3)
    det_err = np.max(np.abs(detR - 1.0)) if detR.size else 0.0

    RtR = np.einsum("...ik,...jk->...ij", R, R)
    I = np.eye(3)[None, :, :]
    ortho_err = np.max(np.linalg.norm(RtR - I, axis=(1, 2))) if RtR.size else 0.0
    ortho_ok = ortho_err < 1e-3

    print(f"  • {name:20s} | last_row_ok={last_row_ok} | det≈1 ok={det_ok} (max|det-1|={fmt(det_err)}) | ortho_err_max={fmt(ortho_err)}")
    if not last_row_ok and bad_last.size:
        print(f"    ↳ bad last-row count: {bad_last.size} / {Ts.shape[0]} (showing first 5): {bad_last[:5]}")

def main():
    link_count = int(input("How many links do you want to check?: ").strip())
    data_dir = os.path.join("data_storage", f"link_{link_count}")

    # 파일명 감지: partial 우선 존재하면 그걸, 아니면 정식 파일
    cand = [os.path.join(data_dir, "fault_dataset_partial.npz"),
            os.path.join(data_dir, "fault_dataset.npz")]
    data_path = next((p for p in cand if os.path.exists(p)), None)
    if data_path is None:
        raise FileNotFoundError(f"❌ {cand} 중 존재하는 파일이 없습니다.")

    data = np.load(data_path, allow_pickle=False)
    print(f"✅ Data loaded from: {data_path}")
    print("📦 keys:", list(data.files))

    # 필수/선택 키
    keys_expect_shapes_hint = {
        "desired_ee": "(S, T, 4, 4)",
        "actual_ee": "(S, T, 4, 4)",
        "desired_link_rel": "(S, T, L, 4, 4)",
        "actual_link_rel": "(S, T, L, 4, 4)",
        "desired_link_cum": "(S, T, L, 4, 4)",
        "actual_link_cum": "(S, T, L, 4, 4)",
        "label": "(S, T, 8*L)",
        "which_fault_mask": "(S, ?)",
        "onset_idx": "(S, ?)",
        "t0": "(S,)",
        "timestamps": "(T,)",
        "dt": "()",
        "link_count": "()",
        "dof": "()",
        "joint_counts": "(S, L)",
    }

    # 1) 기본 무결성: NaN/Inf, 범위
    total_nan = total_inf = 0
    for k in data.files:
        arr = data[k]
        if is_numeric_array(arr):
            n, f = check_nonfinite(k, arr)
            total_nan += n
            total_inf += f
        else:
            print(f"  • {k:20s} | non-numeric, type={type(arr)}")
    print(f"\n[Summary] total NaN={total_nan} | total Inf={total_inf}\n")

    # 2) 구조/치수 일관성 체크
    #    T, L, S 추정
    T = int(data["timestamps"].shape[0]) if "timestamps" in data else None
    L = int(data["link_count"]) if "link_count" in data else link_count
    # S는 desired_ee 첫 차원으로 추정
    S = None
    if "desired_ee" in data:
        S = int(data["desired_ee"].shape[0])
    elif "actual_ee" in data:
        S = int(data["actual_ee"].shape[0])
    elif "label" in data:
        S = int(data["label"].shape[0])

    print("📐 inferred sizes ->", f"S={S}", f"T={T}", f"L={L}")

    def expect(cond, msg):
        if not cond:
            print(f"  [MISMATCH] {msg}")
        else:
            print(f"  [OK] {msg}")

    if "desired_ee" in data:
        expect(data["desired_ee"].ndim == 4 and data["desired_ee"].shape[-2:] == (4,4),
               f"desired_ee has shape {data['desired_ee'].shape} ≈ (S, T, 4, 4)")
        if T is not None:
            expect(data["desired_ee"].shape[1] == T, f"desired_ee.T == timestamps.T ({data['desired_ee'].shape[1]} vs {T})")

    if "actual_ee" in data:
        expect(data["actual_ee"].ndim == 4 and data["actual_ee"].shape[-2:] == (4,4),
               f"actual_ee has shape {data['actual_ee'].shape} ≈ (S, T, 4, 4)")
        if T is not None:
            expect(data["actual_ee"].shape[1] == T, f"actual_ee.T == timestamps.T ({data['actual_ee'].shape[1]} vs {T})")

    def check_link_blocks(name):
        if name not in data: 
            return
        arr = data[name]
        expect(arr.ndim == 5 and arr.shape[-2:] == (4,4),
               f"{name} has shape {arr.shape} ≈ (S, T, L, 4, 4)")
        if T is not None:
            expect(arr.shape[1] == T, f"{name}.T == timestamps.T ({arr.shape[1]} vs {T})")
        expect(arr.shape[2] == L, f"{name}.L matches link_count ({arr.shape[2]} vs {L})")

    for k in ["desired_link_rel","actual_link_rel","desired_link_cum","actual_link_cum"]:
        check_link_blocks(k)

    if "label" in data:
        arr = data["label"]
        expect(arr.ndim == 3, f"label has shape {arr.shape} ≈ (S, T, 8*L)")
        if T is not None:
            expect(arr.shape[1] == T, f"label.T == timestamps.T ({arr.shape[1]} vs {T})")
        if L is not None:
            expect(arr.shape[2] == 8*L, f"label width == 8*L ({arr.shape[2]} vs {8*L})")
        # label 값 분포
        uniq = np.unique(arr)
        print(f"  • label unique values: {uniq}")

    if "joint_counts" in data:
        jc = data["joint_counts"]
        expect(jc.ndim == 2 and jc.shape[1] == L,
               f"joint_counts shape {jc.shape} ≈ (S, L) with link_count={L}")

    # 3) SE(3) 유효성(회전정규성/마지막행) 검사
    if "desired_ee" in data:      check_se3_block("desired_ee", data["desired_ee"])
    if "actual_ee" in data:       check_se3_block("actual_ee", data["actual_ee"])
    if "desired_link_rel" in data:check_se3_block("desired_link_rel", data["desired_link_rel"])
    if "actual_link_rel" in data: check_se3_block("actual_link_rel", data["actual_link_rel"])
    if "desired_link_cum" in data:check_se3_block("desired_link_cum", data["desired_link_cum"])
    if "actual_link_cum" in data: check_se3_block("actual_link_cum", data["actual_link_cum"])

    print("\n✅ Integrity check finished.")

if __name__ == "__main__":
    main()
