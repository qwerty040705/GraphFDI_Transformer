# parameters.py
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List

# ---------- Utility ----------
def skew(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3,)
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]], dtype=float)

def Ad(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]; p = T[:3, 3]
    return np.block([[R, np.zeros((3, 3))], [skew(p) @ R, R]])

def get_inertia_tensor(mass: float, joint_to_com: np.ndarray, inertia_matrix: np.ndarray) -> np.ndarray:
    T_ba = np.eye(4, dtype=float); T_ba[:3, 3] = -np.asarray(joint_to_com, dtype=float).reshape(3,)
    Ad_T = Ad(T_ba)
    upper = np.hstack((inertia_matrix, np.zeros((3, 3), dtype=float)))
    lower = np.hstack((np.zeros((3, 3), dtype=float), mass * np.eye(3, dtype=float)))
    M = np.vstack((upper, lower))
    return Ad_T.T @ M @ Ad_T

def calculate_anti_windup(Kp_diag: np.ndarray, factor: float) -> np.ndarray:
    Kp_diag = np.asarray(Kp_diag, dtype=float).reshape(-1,)
    reverse = 1.0 / np.clip(Kp_diag, 1e-8, None)
    reverse *= float(np.clip(factor, 1/3, 3.0))
    reverse = np.clip(reverse, 0.0, 1.0)
    return np.diag(reverse)

# ---------- Data ----------
@dataclass
class PIDGains:
    p: np.ndarray
    i: np.ndarray
    d: np.ndarray
    a: Optional[np.ndarray] = None
    p_length_max: Optional[float] = None
    i_force_max: Optional[np.ndarray] = None
    i_torque_max: Optional[np.ndarray] = None

@dataclass
class ODAR:
    mass: float
    length: float
    joint_to_com: np.ndarray
    inertia_matrix: np.ndarray
    position_PID: PIDGains
    orientation_PID: PIDGains
    max_thrust: float
    B: np.ndarray
    Bnsv: Dict[str, np.ndarray]
    body_joint_screw_axes: List[np.ndarray]
    joint_inertia_tensor: List[np.ndarray]

# ---------- Main ----------
def get_parameters(nlinks: int) -> Dict:
    """
    nlinks(입력 링크 개수)대로만 모듈을 생성.
    Link 0: 3 DOF (Az, Ay, Ax)
    Link 1..: 2 DOF (Ay, Ax)
    """
    if nlinks < 1:
        raise ValueError("nlinks must be >= 1")

    params: Dict = {}
    params['LASDRA'] = {'total_link_number': int(nlinks)}

    # 기본 상수
    default_mass = 1.6
    default_length = 1.0
    joint_to_com = 0.5 * default_length * np.array([1.0, 0.0, 0.0], dtype=float)
    Ixx, Iyy, Izz = 0.015, 0.1, 0.1
    inertia_matrix = np.diag([Ixx, Iyy, Izz]).astype(float)
    max_thrust = 100000

    B = np.array([
        [0.6797,  0.6797,  0.6797,  0.6797,  0.6797,  0.6797,  0.6797,  0.6797],
        [0.1908,  0.1908, -0.1908, -0.1908,  0.1908,  0.1908, -0.1908, -0.1908],
        [0.7082, -0.7082,  0.7082, -0.7082,  0.7082, -0.7082,  0.7082, -0.7082],
        [-0.1026, -0.1026, -0.1026, -0.1026,  0.1026,  0.1026,  0.1026,  0.1026],
        [-0.0711,  0.1980, -0.1980,  0.0711,  0.0711, -0.1980,  0.1980, -0.0711],
        [0.0894, -0.0169,  0.0169, -0.0894, -0.0894,  0.0169, -0.0169,  0.0894]
    ], dtype=float)
    Bnsv = {'upper': np.array([1, -1, -1, 1, 0, 0, 0, 0], dtype=float),
            'lower': np.array([0, 0, 0, 0, 1, -1, -1, 1], dtype=float)}

    pos_p = np.diag([5.0, 3.0, 8.0]).astype(float)
    pos_i = np.diag([2.5, 2.0, 4.0]).astype(float)
    pos_d = np.diag([4.0, 2.0, 7.0]).astype(float)
    pos_a = calculate_anti_windup(np.diag(pos_p), 3)
    position_PID = PIDGains(p=pos_p, i=pos_i, d=pos_d, a=pos_a,
                            p_length_max=0.6, i_force_max=np.array([15.0, 5.0, 20.0], dtype=float))
    ori_p = np.diag([0.40, 1.20, 1.20]).astype(float)
    ori_i = np.diag([0.20, 0.30, 0.30]).astype(float)
    ori_d = np.diag([0.20, 0.90, 0.90]).astype(float)
    ori_a = calculate_anti_windup(np.diag(ori_p), 3)
    orientation_PID = PIDGains(p=ori_p, i=ori_i, d=ori_d, a=ori_a,
                               i_torque_max=np.array([1.5, 2.0, 2.0], dtype=float))

    Ax = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    Ay = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    Az = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=float)

    odars: List[ODAR] = []
    for i in range(nlinks):
        if i == 0:
            screw_axes = [Az, Ay, Ax]
            inertia_yaw   = np.diag([0.0, 0.0, Izz]).astype(float)
            inertia_pitch = np.diag([0.0, Iyy, 0.0]).astype(float)
            inertia_roll  = np.diag([Ixx, 0.0, 0.0]).astype(float)
            tensors = [
                get_inertia_tensor(0.0,          joint_to_com, inertia_yaw),
                get_inertia_tensor(0.0,          joint_to_com, inertia_pitch),
                get_inertia_tensor(default_mass, joint_to_com, inertia_roll),
            ]
        else:
            screw_axes = [Ay, Ax]
            inertia_pitch = np.diag([0.0, Iyy, 0.0]).astype(float)
            inertia_roll  = np.diag([Ixx, 0.0, 0.0]).astype(float)
            tensors = [
                get_inertia_tensor(0.0,          joint_to_com, inertia_pitch),
                get_inertia_tensor(default_mass, joint_to_com, inertia_roll),
            ]
        odars.append(ODAR(
            mass=default_mass, length=default_length,
            joint_to_com=joint_to_com, inertia_matrix=inertia_matrix,
            position_PID=position_PID, orientation_PID=orientation_PID,
            max_thrust=max_thrust, B=B, Bnsv=Bnsv,
            body_joint_screw_axes=screw_axes, joint_inertia_tensor=tensors
        ))

    params['ODAR'] = odars

    # 플랫한 리스트로 기록
    screw_axes_all: List[np.ndarray] = []
    inertia_all: List[np.ndarray] = []
    for odar in params['ODAR']:
        screw_axes_all.extend(odar.body_joint_screw_axes)
        inertia_all.extend(odar.joint_inertia_tensor)

    params['LASDRA']['body_joint_screw_axes'] = screw_axes_all
    params['LASDRA']['inertia_matrix'] = inertia_all
    params['LASDRA']['dof'] = len(screw_axes_all)
    return params
