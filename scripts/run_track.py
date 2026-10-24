import argparse
import yaml
from datetime import datetime
from src.pipeline import run_pipeline

def parse_args():
    p = argparse.ArgumentParser(description="BadmintonSight shuttlecock tracker")
    p.add_argument("--config", type=str, default="configs/track_cfg.yaml")
    p.add_argument("--video", type=str, required=True)
    p.add_argument("--weights", type=str, required=True, help="yolo pt path")
    p.add_argument("--output", type=str, default="")
    return p.parse_args()

def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["model_path"] = args.weights
    if not args.output:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = f"outputs/track_{ts}.mp4"
    else:
        out = args.output
    run_pipeline(args.video, out, cfg)

if __name__ == "__main__":
    main()
