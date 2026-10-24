import numpy as np

def ransac_clean_parabola(traj, min_samples=5, max_iter=80, residual_thresh=12):
    if len(traj) < min_samples:
        return traj
    pts = np.array(traj, dtype=np.float32)
    best_mask = np.zeros(len(pts), dtype=bool)
    np.random.seed(42)
    for _ in range(max_iter):
        idx = np.random.choice(len(pts), min_samples, replace=False)
        sample = pts[idx]
        try:
            x = sample[:,0]
            y = sample[:,1]
            X = np.vstack([x*x, x, np.ones(len(x))]).T
            coeff, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            a,b,c = coeff
        except Exception:
            continue
        y_pred = a * pts[:,0]**2 + b * pts[:,0] + c
        res = np.abs(y_pred - pts[:,1])
        mask = res < residual_thresh
        if np.sum(mask) > np.sum(best_mask):
            best_mask = mask
    cleaned = pts[best_mask].tolist()
    return [tuple(p) for p in cleaned]
