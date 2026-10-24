import cv2
import numpy as np

def draw_fade_points(frame, points_list, max_keep=30):
    pts = points_list[-max_keep:]
    n = len(pts)
    if n < 1:
        return
    for idx, (x, y) in enumerate(pts):
        alpha = ((idx + 1) / n) ** 1.6
        c = int(255 * alpha)
        color = (c, c, 255)
        r = int(3 + alpha * 3)
        overlay = frame.copy()
        cv2.circle(overlay, (int(x), int(y)), r, color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)

def draw_candidates(frame, candidates):
    for cand in candidates:
        px, py = cand["point"]
        color = (255,0,255) if cand.get("is_static") else (255,0,0)
        cv2.circle(frame, (px, py), 4, color, -1)
