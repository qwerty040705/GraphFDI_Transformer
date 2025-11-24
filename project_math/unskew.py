import numpy as np

def unskew(R):
    Rv = np.array([
        R[2, 1],
        R[0, 2],
        R[1, 0]
    ])
    return Rv
