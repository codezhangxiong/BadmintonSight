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

from src.pipeline import run_analysis_only, run_pipeline, run_render_only  # noqa: E402


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
    p = argparse.ArgumentParser(
        description="BadmintonSight shuttlecock tracker（两阶段：先分析，再渲染）"
    )
    p.add_argument("--config", type=str, default="configs/track_cfg.yaml",
                   help="配置 yaml（相对路径以项目根为基准）")
    p.add_argument("--video", type=str, required=True, help="输入视频路径")
    p.add_argument("--weights", type=str, default="", help="YOLO .pt 权重路径（--render-from 时不需要）")
    p.add_argument("--output", type=str, default="",
                   help="输出 mp4；默认 outputs/track_<时间戳>.mp4")
    p.add_argument("--no-gui", action="store_true",
                   help="不弹 OpenCV 窗口（服务器/CI/批处理用）")
    p.add_argument("--no-traj", action="store_true", help="不写轨迹 CSV")
    p.add_argument("--limit", type=int, default=None, help="只处理前 N 帧（调试用）")
    p.add_argument("--analyze-only", action="store_true",
                   help="只跑分析阶段（写 CSV + analysis.json），不渲染视频")
    p.add_argument("--render-from", type=str, default="",
                   help="从已有 analysis.json 渲染视频（跳过 YOLO，仍需 --video）")
    p.add_argument("--analysis-json", type=str, default="",
                   help="另存分析结果 JSON；--analyze-only 默认 <输出主干>.analysis.json")
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


def _default_out_path():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(PROJECT_ROOT / "outputs" / f"track_{ts}.mp4")


def main():
    args = parse_args()
    video_path = _resolve(args.video, must_exist=True)
    out_path = str(_resolve(args.output)) if args.output else _default_out_path()

    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit 必须是正整数")

    analysis_json = str(_resolve(args.analysis_json)) if args.analysis_json else ""

    if args.render_from:
        # 只渲染：不需要权重，也不需要（重新）分析
        render_json = _resolve(args.render_from, must_exist=True)
        summary = run_render_only(
            str(video_path), out_path, str(render_json), show_gui=not args.no_gui
        )
        config_path = "(未使用：render-only)"
        weights_path = "(未使用：render-only)"
    else:
        if not args.weights:
            raise ValueError("缺少 --weights（只有 --render-from 模式可以省略权重）")
        config_path = _resolve(args.config, must_exist=True)
        weights_path = _resolve(args.weights, must_exist=True)
        cfg = load_config(config_path)
        cfg["model_path"] = str(weights_path)

        if args.analyze_only:
            summary = run_analysis_only(
                str(video_path),
                cfg,
                out_path,
                limit=args.limit,
                save_traj=not args.no_traj,
                analysis_json=analysis_json or None,
            )
        else:
            summary = run_pipeline(
                str(video_path),
                out_path,
                cfg,
                show_gui=not args.no_gui,
                limit=args.limit,
                save_traj=not args.no_traj,
                analysis_json=analysis_json or None,
            )

    print("=" * 60)
    print(f"视频      : {video_path}")
    print(f"权重      : {weights_path}")
    print(f"配置      : {config_path}")
    if summary.get("rendered_frames"):
        print(f"输出视频  : {summary['out_path']}（{summary['rendered_frames']} 帧）")
    else:
        print(f"输出视频  : （未渲染，--analyze-only）")
    if summary.get("traj_csv"):
        print(f"逐帧轨迹  : {summary['traj_csv']}")
        print(f"分段汇总  : {summary['segments_csv']}")
        print(f"清洗轨迹  : {summary['clean_csv']}")
    if summary.get("analysis_json"):
        print(f"分析结果  : {summary['analysis_json']}")
    print(
        f"处理帧数  : {summary['frames']} / {summary['expected_frames']}"
        f"  （接受 {summary['accepted']}，离群丢弃 {summary['rejected']}，无观测 {summary['miss']}）"
    )
    print(f"轨迹段数  : {len(summary['segments'])}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
