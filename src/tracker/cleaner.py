import numpy as np


def ransac_clean_parabola(traj, min_samples=5, max_iter=80, residual_thresh=12, return_mask=False):
    """RANSAC 拟合 y = a*x^2 + b*x + c 并返回内点。

    return_mask=True 时返回 (cleaned_points, mask)，mask 是与输入等长的布尔数组，
    调用方可用它把内点映射回原始帧号等附加信息（pipeline 依赖此行为落盘清洗后轨迹）。
    """
    if len(traj) < min_samples:
        if return_mask:
            return list(traj), np.ones(len(traj), dtype=bool)
        return traj
    pts = np.array(traj, dtype=np.float32)
    best_mask = np.zeros(len(pts), dtype=bool)
    np.random.seed(42)
    for _ in range(max_iter):
        idx = np.random.choice(len(pts), min_samples, replace=False)
        sample = pts[idx]
        try:
            x = sample[:, 0]
            y = sample[:, 1]
            X = np.vstack([x * x, x, np.ones(len(x))]).T
            coeff, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            a, b, c = coeff
        except Exception:
            continue
        y_pred = a * pts[:, 0] ** 2 + b * pts[:, 0] + c
        res = np.abs(y_pred - pts[:, 1])
        mask = res < residual_thresh
        if np.sum(mask) > np.sum(best_mask):
            best_mask = mask
    cleaned = pts[best_mask].tolist()
    cleaned = [tuple(p) for p in cleaned]
    if return_mask:
        return cleaned, best_mask
    return cleaned
