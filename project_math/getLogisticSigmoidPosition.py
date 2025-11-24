import numpy as np

def getLogisticSigmoidPosition(s_end, t_end, t_query, t_d=0.5):
    a = 5 

    if 2 * t_d > t_end:
        raise ValueError("improper calculation conducted: t_d * 2 > t_end")

    s_end = np.atleast_2d(s_end)
    if s_end.shape[0] < s_end.shape[1]:
        s_end = s_end.T  
    n = s_end.shape[0]

    t_query = np.atleast_1d(t_query).flatten()
    T = t_query.shape[0]

    v_0 = s_end / (t_end - t_d) 
    x_query = np.zeros((n, T))

    idx1 = t_query <= t_d
    idx2 = (t_query > t_d) & (t_query < (t_end - t_d))
    idx3 = t_query >= (t_end - t_d)

    if np.any(idx1):
        tq = t_query[idx1]
        term = 0.25 * t_d / a * np.log(np.cosh(a * (2 * tq / t_d - 1)) / np.cosh(a)) + 0.5 * tq
        x_query[:, idx1] = v_0 @ term.reshape(1, -1) 

    if np.any(idx2):
        tq = t_query[idx2]
        x_query[:, idx2] = 0.5 * v_0 * t_d + v_0 @ (tq - t_d).reshape(1, -1)

    if np.any(idx3):
        tq = t_query[idx3]
        term = 0.25 * t_d / a * np.log(np.cosh(a * (2 * (tq - t_end + t_d) / t_d - 1)) / np.cosh(a)) - 0.5 * (tq - t_end + t_d)
        x_query[:, idx3] = s_end - 0.5 * v_0 * t_d - v_0 @ term.reshape(1, -1)

    return x_query
