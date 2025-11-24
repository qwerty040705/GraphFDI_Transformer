import numpy as np
from .skew import skew 

def Ad(T):
    R = T[:3, :3]
    p = T[:3, 3]
    Ad_T = np.block([
        [R, np.zeros((3, 3))],
        [skew(p) @ R, R]
    ])
    return Ad_T
