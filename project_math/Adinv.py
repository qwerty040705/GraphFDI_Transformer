import numpy as np
from .skew import skew

def Adinv(T):
    R = T[:3, :3]
    p = T[:3, 3]
    Ad_Tinv = np.block([
        [R.T, np.zeros((3, 3))],
        [-R.T @ skew(p), R.T]
    ])
    return Ad_Tinv
