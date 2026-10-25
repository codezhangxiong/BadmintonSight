import argparse
import sys
from datetime import datetime
from pathlib import Path

# 允许从任意工作目录直接运行 `python scripts/run_track.py`：
# 把项目根加入 sys.path 后，src/ 与 utils/ 才能被导入（必须在 import src.* 之前执行）。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml  # noqa: E402

from src.pipeline import run_pipeline  # noqa: E402


def _resolve(path_str, must_exist=False):
    """相对路径统一按项目根解析，避免依赖当前工作目录。"""
    p = Path(path_str)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    p = p.resolve()
    if must_exist and not p.exists():
        raise FileNotFoundError(f"找不到文件: {p}")
    return p


def parse_args():
    p = argparse.ArgumentParser(description="BadmintonSight shuttlecock tracker")
    p.add_argument("--config", type=str, default="configs/track_cfg.yaml",
                   help="配置 yaml（相对路径以项目根为基准）")
    p.add_argument("--video", type=str, required=True, help="输入视频路径")
    p.add_argument("--weights", type=str, required=True, help="YOLO .pt 权重路径")
    p.add_argument("--output", type=str, default="",
                   help="输出 mp4；默认 outputs/track_<时间戳>.mp4")
    p.add_argument("--no-gui", action="store_true",
                   help="不弹 OpenCV 窗口（服务器/CI/批处理用）")
    p.add_argument("--no-traj", action="store_true", help="不写轨迹 CSV")
    p.add_argument("--limit", type=int, default=None, help="只处理前 N 帧（调试用）")
    return p.parse_args()


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"配置文件内容为空或不是字典: {config_path}")
    for key in ("tracker", "detector", "ransac"):
        if key not in cfg:
            raise ValueError(f"配置文件缺少 '{key}' 段: {config_path}")
    return cfg


def main():
    args = parse_args()
    config_path = _resolve(args.config, must_exist=True)
    video_path = _resolve(args.video, must_exist=True)
    weights_path = _resolve(args.weights, must_exist=True)

    cfg = load_config(config_path)
    cfg["model_path"] = str(weights_path)

    if args.output:
        out_path = str(_resolve(args.output))
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = str(PROJECT_ROOT / "outputs" / f"track_{ts}.mp4")

    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit 必须是正整数")

    summary = run_pipeline(
        str(video_path),
        out_path,
        cfg,
        show_gui=not args.no_gui,
        limit=args.limit,
        save_traj=not args.no_traj,
    )

    print("=" * 60)
    print(f"视频      : {video_path}")
    print(f"权重      : {weights_path}")
    print(f"配置      : {config_path}")
    print(f"输出视频  : {summary['out_path']}")
    if summary.get("traj_csv"):
        print(f"逐帧轨迹  : {summary['traj_csv']}")
        print(f"分段汇总  : {summary['segments_csv']}")
        print(f"清洗轨迹  : {summary['clean_csv']}")
    print(
        f"处理帧数  : {summary['frames']} / {summary['expected_frames']}"
        f"  （接受 {summary['accepted']}，离群丢弃 {summary['rejected']}，无观测 {summary['miss']}）"
    )
    print(f"轨迹段数  : {len(summary['segments'])}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
