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


def extract_shuttlecock_candidates(yolo_boxes, frame_shape, cfg):
    """把 YOLO 检测框转成候选点。

    保留的过滤：置信度下限（conf_min）、无效框保护（宽或高 <= 0）。
    **已移除**的两类门控：
    - **外形门控**（面积比 max_box_area_ratio、长宽比 max_aspect_ratio）：
      本题权重输出的球框恒定约 18×18px，这两个阈值从未真正触发；
      而且它们属于"训练分布假设"，换机位/分辨率或换权重时会误杀有效检测，
      因此改为不做形状/尺寸过滤。
    - **ROI 门控**：原先"画面内缩 50px 再按 ROI 尺寸外扩 8%"，外扩量（94px）
      大于内缩量（50px），最终范围覆盖整幅画面，实际恒为真；且 ROI 硬编码在
      调用方，无法配置。与其保留一个永远为真的开关，不如删除。

    area_ratio / aspect_ratio 仍作为候选元数据返回（关联代价函数会用到），
    但不再参与候选的接受/拒绝。
    """
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
        candidates.append(
            {
                "point": (int(cx), int(cy)),
                "confidence": float(conf),
                # 元数据：供关联代价使用；不再用于过滤
                "area_ratio": area_ratio,
                "aspect_ratio": ar,
                # 以下字段用于轨迹落盘与后续调试
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
