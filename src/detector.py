def extract_shuttlecock_candidates(yolo_boxes, frame_shape, cfg, roi_corners=None):
    boxes = yolo_boxes
    if boxes is None or boxes.xywh.shape[0] < 1:
        return []
    xywh = boxes.xywh.detach().cpu().numpy()
    confs = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else np.ones(len(xywh))
    h, w = frame_shape[:2]
    frame_area = max(1, h * w)
    det_cfg = cfg["detector"]
    candidates = []
    for box, conf in zip(xywh, confs):
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
            (x1,y1),(x2,y2) = roi_corners
            pad = int(max(x2-x1, y2-y1) * det_cfg["roi_padding_ratio"])
            if not ((x1-pad) <= pt[0] <= (x2+pad) and (y1-pad) <= pt[1] <= (y2+pad)):
                in_roi = False
        if not in_roi:
            continue
        candidates.append({
            "point": pt,
            "confidence": float(conf),
            "area_ratio": area_ratio,
            "aspect_ratio": ar
        })
    return candidates
