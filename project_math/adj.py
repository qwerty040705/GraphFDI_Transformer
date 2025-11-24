import numpy as np
from .skew import skew

def adj(V):
    V = np.asarray(V)
    
    w = V[0:3].reshape(3)
    v = V[3:6].reshape(3)

    Ad_V = np.block([
        [skew(w), np.zeros((3, 3))],
        [skew(v), skew(w)]
    ])
    return Ad_V
