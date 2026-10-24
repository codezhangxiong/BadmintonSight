import numpy as np
from utils.common import distance

def predict_linear(history_pts, n_lookback=4):
    if len(history_pts) < n_lookback:
        return None
    arr = np.array(history_pts[-n_lookback:])
    vx = np.mean(np.diff(arr[:, 0]))
    vy = np.mean(np.diff(arr[:, 1]))
    lx, ly = arr[-1]
    return (lx + vx, ly + vy)

def get_current_speed(traj, lookback=4):
    if len(traj) < lookback:
        return 0.0
    arr = np.array(traj[-lookback:])
    diffs = np.hypot(np.diff(arr[:,0]), np.diff(arr[:,1]))
    return float(np.mean(diffs))

def dynamic_step_threshold(traj, base_gain=2.0, min_thresh=60, max_thresh=320, lookback=5):
    if len(traj) < lookback:
        return 180.0
    arr = np.array(traj[-lookback:])
    diffs = np.hypot(np.diff(arr[:,0]), np.diff(arr[:,1]))
    mean_step = np.mean(diffs)
    th = mean_step * base_gain
    return float(np.clip(th, min_thresh, max_thresh))
