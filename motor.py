
import numpy as np

def save_motor_geom(L: int, out_prefix: str = "motor_geom_link"):

    M = 8  

    f_dir_single = np.array([
        [ 0.68,  0.28, -0.68],
        [ 0.68, -0.28,  0.68],
        [-0.68,  0.28,  0.68],
        [-0.68, -0.28, -0.68],
        [ 0.28,  0.68, -0.68],
        [ 0.28, -0.68,  0.68],
        [-0.28,  0.68,  0.68],
        [-0.28, -0.68, -0.68],
    ], dtype=np.float32)

    # 단위벡터로 정규화
    f_dir_single = f_dir_single / np.linalg.norm(f_dir_single, axis=1, keepdims=True)

    r_single = np.array([
        [ 0.40,  0.17, -0.17],
        [ 0.40, -0.17,  0.17],
        [-0.40,  0.17,  0.17],
        [-0.40, -0.17, -0.17],
        [ 0.17,  0.40, -0.17],
        [ 0.17, -0.40,  0.17],
        [-0.17,  0.40,  0.17],
        [-0.17, -0.40, -0.17],
    ], dtype=np.float32)

    rotvec_single = np.zeros((M, 3), dtype=np.float32)

    arm_single = np.ones((M,), dtype=np.float32) * 0.12

    rotvec = np.stack([rotvec_single] * L, axis=0)   # (L, 8, 3)
    r_mL   = np.stack([r_single]       * L, axis=0)  # (L, 8, 3)
    f_dir  = np.stack([f_dir_single]   * L, axis=0)  # (L, 8, 3)
    arm    = np.stack([arm_single]     * L, axis=0)  # (L, 8)


    out_name = f"{out_prefix}{L}.npz"

    np.savez(
        out_name,
        rotvec_m2l=rotvec,
        r_mL=r_mL,
        f_dir=f_dir,
        arm=arm,
    )

    print(f"Saved {out_name} (links={L}, motors/link={M})")


if __name__ == "__main__":
    L = int(input("How many links? : "))
    save_motor_geom(L)
