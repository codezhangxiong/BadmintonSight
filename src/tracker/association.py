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

def select_best_candidate(candidates, ref_pt, traj, cfg):
    if not candidates:
        return None
    from src.tracker.predictor import get_current_speed
    speed = get_current_speed(traj)
    fast_th = cfg["tracker"]["fast_speed_thresh"]
    if ref_pt is None:
        lam = 0.5
    else:
        lam = cfg["tracker"]["lambda_fast"] if speed > fast_th else cfg["tracker"]["lambda_slow"]
    best = None
    minc = float("inf")
    sp = cfg["tracker"]["static_penalty"]
    for cand in candidates:
        tc, cm, ca = compute_cost(cand, ref_pt, lam, static_penalty=sp)
        if tc < minc:
            minc = tc
            best = cand
    return best
