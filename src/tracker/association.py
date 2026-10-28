from utils.common import distance

static_candidate_memory = {}

def mark_static_candidates(candidates, frame_idx, grid_size=10, static_frame_count=3):
    out = []
    for cand in candidates:
        x, y = cand["point"]
        gx = round(x / grid_size)
        gy = round(y / grid_size)
        key = (gx, gy)
        if key not in static_candidate_memory:
            static_candidate_memory[key] = {"pts": [], "last_frame": -999}
        rec = static_candidate_memory[key]
        if frame_idx - rec["last_frame"] == 1:
            rec["pts"].append((x,y))
        else:
            rec["pts"] = [(x,y)]
        rec["last_frame"] = frame_idx
        if len(rec["pts"]) > static_frame_count:
            rec["pts"] = rec["pts"][-static_frame_count:]
        new_cand = cand.copy()
        new_cand["is_static"] = len(rec["pts"]) >= static_frame_count
        out.append(new_cand)
    drop_keys = [k for k,v in static_candidate_memory.items() if frame_idx - v["last_frame"] > 10]
    for k in drop_keys:
        del static_candidate_memory[k]
    return out

def compute_cost(cand, ref_pt, lam, static_penalty=120.0):
    cost_app = (- cand["confidence"]) + cand["area_ratio"] * 3000.0
    if ref_pt is None:
        return cost_app, 0.0, cost_app
    d = distance(cand["point"], ref_pt)
    sp = static_penalty if cand.get("is_static", False) else 0.0
    cost_mot = d + sp
    total = lam * cost_mot + (1 - lam) * cost_app
    return total, cost_mot, cost_app

def _rank_candidates(candidates, ref_pt, traj, cfg):
    """按关联代价升序排序，返回 [(total_cost, candidate), ...]。"""
    from src.tracker.predictor import get_current_speed

    track_cfg = cfg["tracker"]
    speed = get_current_speed(traj)
    fast_th = track_cfg["fast_speed_thresh"]
    if ref_pt is None:
        lam = 0.5
    else:
        lam = track_cfg["lambda_fast"] if speed > fast_th else track_cfg["lambda_slow"]
    sp = track_cfg["static_penalty"]
    scored = []
    for cand in candidates:
        total, _, _ = compute_cost(cand, ref_pt, lam, static_penalty=sp)
        scored.append((total, cand))
    scored.sort(key=lambda item: item[0])
    return scored


def select_best_candidate(candidates, ref_pt, traj, cfg, last_obs=None):
    """选一帧的关联目标。

    基础规则：取关联代价最小的候选。

    方案 C（`gates.association` 开启时，默认开启）：
    当 top-1 与 top-2 的代价差小于 `tracker.assoc_margin` 时，说明两个候选
    "都说得通"（典型场景：球与反光点在两组位置交替出现，平滑后的预测点落在
    两者之间）。此时按以下顺序裁决：
      1. **连续性优先**：若次优候选到上一个已接受观测的距离明显更近
         （`db + assoc_cont_margin < da`），选它 —— 球不会无理由瞬移；
      2. **距离接近时用置信度**：若 `|da - db| <= assoc_cont_margin`，选 conf 更高者；
      3. 否则保持代价最小。
    若没有上次观测（`last_obs=None`，如段首帧），直接按 conf 决胜。

    注意：不能用"conf 高者恒胜"——实测存在高 conf 假阳性（如球拍/反光点，
    conf≥0.8 但偏离轨迹 100+px），它会把平滑窗口带偏并引发连锁拒绝。
    连续性（到上一观测的距离）才是更稳的裁决依据。

    可选：`tracker.assoc_max_dist > 0` 时把离预测点超过该距离的候选从候选池
    剔除；若剔除后为空则保留全部（不硬拒绝，避免快球误丢）。

    关闭 `gates.association` 时退化为"纯代价最小"（旧行为，用于消融对照）。
    """
    if not candidates:
        return None

    scored = _rank_candidates(candidates, ref_pt, traj, cfg)
    if not cfg.get("gates", {}).get("association", True):
        return scored[0][1]

    track_cfg = cfg["tracker"]
    margin = float(track_cfg.get("assoc_margin", 0.0) or 0.0)
    cont_margin = float(track_cfg.get("assoc_cont_margin", 30.0) or 30.0)
    max_dist = float(track_cfg.get("assoc_max_dist", 0.0) or 0.0)

    pool = scored
    if max_dist > 0 and ref_pt is not None:
        kept = [
            (total, cand)
            for total, cand in scored
            if distance(cand["point"], ref_pt) <= max_dist
        ]
        if kept:
            pool = kept

    best = pool[0][1]
    if margin > 0 and len(pool) >= 2 and (pool[1][0] - pool[0][0]) < margin:
        second = pool[1][1]
        if last_obs is None:
            if second["confidence"] > best["confidence"]:
                best = second
        else:
            da = distance(best["point"], last_obs)
            db = distance(second["point"], last_obs)
            if db + cont_margin < da:
                best = second
            elif abs(da - db) <= cont_margin and second["confidence"] > best["confidence"]:
                best = second
    return best
