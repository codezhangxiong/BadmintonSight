import numpy as np

def distance(pt_a, pt_b):
    return float(np.hypot(pt_a[0] - pt_b[0], pt_a[1] - pt_b[1]))
