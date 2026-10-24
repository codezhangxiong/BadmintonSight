import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO

from src.tracker.predictor import predict_linear, dynamic_step_threshold
from src.tracker.association import mark_static_candidates, select_best_candidate
from src.tracker.cleaner import ransac_clean_parabola
from src.detector import extract_shuttlecock_candidates
from src.visualizer import draw_fade_points, draw_candidates
from utils.common import distance

def run_pipeline(video_path, out_path, cfg):
    model = YOLO(cfg["model_path"])
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    track_cfg = cfg["tracker"]
    ransac_cfg = cfg["ransac"]
    smooth_win = deque(maxlen=track_cfg["smooth_window_len"])
    full_traj = []
    missing = 0
    MAX_MISS = track_cfg["max_miss"]
    roi = [(50,50), (w-50, h-50)]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        fid = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        results = model(frame, conf=cfg["detector"]["yolo_conf"], verbose=False)[0]

        # draw raw boxes for debug
        raw = results.boxes
        if raw is not None:
            xywh = raw.xywh.detach().cpu().numpy()
            confs = raw.conf.detach().cpu().numpy()
            for cx, cy, bw, bh, conf in zip(xywh[:,0],xywh[:,1],xywh[:,2],xywh[:,3], confs):
                x1 = int(cx - bw/2)
                y1 = int(cy - bh/2)
                x2 = int(cx + bw/2)
                y2 = int(cy + bh/2)
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),1)
                cv2.putText(frame,f"{conf:.2f}",(x1,y1-2),cv2.FONT_HERSHEY_SIMPLEX,0.35,(0,0,255),1)

        candidates = extract_shuttlecock_candidates(results.boxes, frame.shape, cfg, roi)
        candidates = mark_static_candidates(
            candidates, fid,
            grid_size=track_cfg["grid_size"],
            static_frame_count=track_cfg["static_frame_count"]
        )
        draw_candidates(frame, candidates)

        ref_pt = predict_linear(full_traj, n_lookback=4)
        best = select_best_candidate(candidates, ref_pt, full_traj, cfg)

        smooth_pt = None
        if best is not None:
            obs_pt = best["point"]
            cv2.circle(frame, obs_pt, 7, (0,255,255), -1)
            outlier = False
            if len(smooth_win) > 0:
                last_p = smooth_win[-1]
                d = distance(obs_pt, last_p)
                th = dynamic_step_threshold(
                    full_traj,
                    base_gain=track_cfg["dynamic_gain"],
                    min_thresh=track_cfg["dynamic_min_thresh"],
                    max_thresh=track_cfg["dynamic_max_thresh"]
                )
                if d > th:
                    outlier = True
            if outlier:
                missing += 1
                print(f"[Frame {fid}] OBS {obs_pt} outlier drop")
            else:
                missing = 0
                smooth_win.append(np.array(obs_pt))
                arr = np.array(list(smooth_win))
                smooth_pt = tuple(np.mean(arr, axis=0))
                full_traj.append(smooth_pt)
                print(f"[Frame {fid}] obs {obs_pt}, smooth {np.round(smooth_pt,1)}")
        else:
            missing += 1
            print(f"[Frame {fid}] no obs, missing={missing}")

        if smooth_pt is not None:
            cv2.circle(frame, (int(smooth_pt[0]), int(smooth_pt[1])),6,(0,0,255),-1)
        draw_fade_points(frame, full_traj, max_keep=30)

        out.write(frame)
        cv2.imshow("debug", frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

        if missing > MAX_MISS:
            if len(full_traj) > ransac_cfg["min_samples"]:
                print(f"\n==== trajectory end, raw len: {len(full_traj)} ====")
                cleaned = ransac_clean_parabola(
                    full_traj,
                    min_samples=ransac_cfg["min_samples"],
                    max_iter=ransac_cfg["max_iter"],
                    residual_thresh=ransac_cfg["residual_thresh"]
                )
                print(f"==== cleaned len: {len(cleaned)} ====\n")
                # cleaned 就是业务可用轨迹
            full_traj.clear()
            smooth_win.clear()
            missing = 0

    cap.release()
    out.release()
    cv2.destroyAllWindows()
