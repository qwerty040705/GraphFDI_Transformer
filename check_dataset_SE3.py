import os
import numpy as np

link_count = int(input("How many links do you want to check?: ").strip())
data_dir = os.path.join("data_storage", f"link_{link_count}")
data_path = os.path.join(data_dir, "fault_dataset.npz")

if not os.path.exists(data_path):
    raise FileNotFoundError(f"{data_path} not found.")

data = np.load(data_path)
desired = data['desired']  # (S, 200, 4, 4)
actual = data['actual']    # (S, 200, 4, 4)
label = data['label']      # (S, 200, 8*N)

S, T, _, _ = desired.shape
motor_count = label.shape[2]

print(f"desired.shape = {desired.shape}")
print(f"actual.shape  = {actual.shape}")
print(f"label.shape   = {label.shape} (motor_count={motor_count})")


raw_times = [0, 1, 10, 50, 100, 150, 200, 250, 500, 750, 800, 999]
time_idx_list = [t for t in raw_times if t < T]

def decompose_se3(T4):
    R = T4[:3, :3]
    p = T4[:3, 3]
    return R, p

def rot_angle(R_err):
    tr = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(tr))

sample_idx = 0
assert 0 <= sample_idx < S, "sample_idx out of range"

for time_idx in time_idx_list:
    print("\n" + "="*50)
    print(f"[Sample {sample_idx}, Time {time_idx}]")

    lbl = label[sample_idx, time_idx].astype(int)
    print("Motor labels:", lbl)

    T_des = desired[sample_idx, time_idx]
    T_act = actual[sample_idx, time_idx]
    R_des, p_des = decompose_se3(T_des)
    R_act, p_act = decompose_se3(T_act)

    print("\nDesired SE(3):\n", T_des)
    print("\nActual  SE(3):\n", T_act)

    pos_err = np.linalg.norm(p_act - p_des)                     # [m]
    R_err   = R_des.T @ R_act                                   # des→act
    ang_err = rot_angle(R_err)                                   # [rad]
    ang_deg = np.degrees(ang_err)

    print(f"\nPosition error ‖p_act - p_des‖ = {pos_err:.6f} m")
    print(f"Orientation error angle         = {ang_err:.6f} rad ({ang_deg:.3f} deg)")

