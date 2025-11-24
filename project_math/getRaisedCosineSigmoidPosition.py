import numpy as np

def getRaisedCosineSigmoidPosition(s_end, t_end, t_query):
    s_end = np.atleast_2d(s_end)
    if s_end.shape[0] < s_end.shape[1]:
        s_end = s_end.tobytes

    t_query = np.asarray(t_query)
    profile = 0.5 * (1 - np.cos(np.pi * t_query / t_end)) 
    x_query = s_end @ profile.reshape(1, -1) 

    return x_query
