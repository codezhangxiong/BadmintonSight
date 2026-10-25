import numpy as np


def _safe_conf(conf_tensor, n):
    """把 conf 转成 numpy；缺失或长度不匹配时回退为全 1，避免 zip 静默截断。"""
    if conf_tensor is None:
        return np.ones(n, dtype=np.float32)
    try:
        arr = conf_tensor.detach().cpu().numpy()
    except AttributeError:
        arr = np.asarray(conf_tensor)
    if arr.shape[0] != n:
        return np.ones(n, dtype=np.float32)
    return arr


def _safe_cls(cls_tensor, n):
    """把类别 id 转成 numpy；缺失或长度不匹配时返回 None。"""
    if cls_tensor is None:
        return None
    try:
        arr = cls_tensor.detach().cpu().numpy().astype(int)
    except (AttributeError, TypeError, ValueError):
        return None
    return arr if arr.shape[0] == n else None


def extract_shuttlecock_candidates(yolo_boxes, frame_shape, cfg, roi_corners=None):
    boxes = yolo_boxes
    if boxes is None or boxes.xywh.shape[0] < 1:
        return []
    xywh = boxes.xywh.detach().cpu().numpy()
    confs = _safe_conf(getattr(boxes, "conf", None), len(xywh))
    classes = _safe_cls(getattr(boxes, "cls", None), len(xywh))
    h, w = frame_shape[:2]
    frame_area = max(1, h * w)
    det_cfg = cfg["detector"]
    candidates = []
    for i, (box, conf) in enumerate(zip(xywh, confs)):
        if conf < det_cfg["conf_min"]:
            continue
        cx, cy, bw, bh = box
        if bw <= 0 or bh <= 0:
            continue
        area_ratio = (bw * bh) / frame_area
        ar = max(bw / bh, bh / bw)
        if area_ratio > det_cfg["max_box_area_ratio"] or ar > det_cfg["max_aspect_ratio"]:
            continue
        pt = (int(cx), int(cy))
        in_roi = True
        if roi_corners is not None:
            (x1, y1), (x2, y2) = roi_corners
            pad = int(max(x2 - x1, y2 - y1) * det_cfg["roi_padding_ratio"])
            if not ((x1 - pad) <= pt[0] <= (x2 + pad) and (y1 - pad) <= pt[1] <= (y2 + pad)):
                in_roi = False
        if not in_roi:
            continue
        candidates.append(
            {
                "point": pt,
                "confidence": float(conf),
                "area_ratio": float(area_ratio),
                "aspect_ratio": float(ar),
                # 以下字段用于轨迹落盘与后续调试，不影响既有过滤/关联逻辑
                "bbox": (
                    int(cx - bw / 2),
                    int(cy - bh / 2),
                    int(cx + bw / 2),
                    int(cy + bh / 2),
                ),
                "cls": int(classes[i]) if classes is not None else None,
            }
        )
    return candidates
