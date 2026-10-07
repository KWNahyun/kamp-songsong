from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path("/home/viplab/contest")
TRAIN = ROOT / "experiments/kamp_synth_train_v1"
EVAL = ROOT / "experiments/kamp_synth_train_v1_eval"
OLD = ROOT / "experiments/kamp_synth_v1_eval"
FROZEN = ROOT / "experiments/kamp_v2_test_frozen"


def main() -> None:
    EVAL.mkdir(parents=True, exist_ok=True)
    (EVAL / "runs").mkdir(exist_ok=True)
    (EVAL / "real").mkdir(exist_ok=True)
    for script in ["infer.py", "analyze_run.py"]:
        destination = EVAL / script
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        destination.symlink_to(OLD / script)

    training = json.loads((TRAIN / "protocol.json").read_text())
    old_protocol = json.loads((FROZEN / "frozen_protocol.json").read_text())
    thresholds = {(job["variant"], job["seed"]): job["val_threshold"] for job in old_protocol["jobs"]}
    jobs = []
    for job in training["jobs"]:
        name = job["name"]
        run = TRAIN / "runs" / name
        if job["family"] == "yolo":
            checkpoint = run / "weights/best.pt"
            config = None
        elif job["family"] == "uq":
            checkpoint = run / "last.pth"
            config = Path(job["config"])
        else:
            checkpoint = run / "best_stg1.pth"
            config = Path(job["config"])
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        jobs.append({
            "name": name,
            "variant": job["variant"],
            "seed": job["seed"],
            "checkpoint": str(checkpoint),
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "config": str(config) if config else None,
            "val_threshold": thresholds[(job["variant"], job["seed"])],
            "threshold_source": "pre-synthetic real-validation operating point; locked for transfer diagnosis",
        })
    protocol = {
        "created": __import__("datetime").datetime.now().astimezone().isoformat(),
        "dataset": str(ROOT / "data/synthetic/kamp_synth_test_v1"),
        "jobs": jobs,
        "training_protocol": str(TRAIN / "protocol.json"),
        "batch": 1,
        "NMS": 0.7,
        "box_scale": 1.0,
        "score_floor": 0.001,
        "threshold_grid": [0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
        "calibration_targets": [0.9, 0.95, 0.98],
        "review_budgets": [0.05, 0.1, 0.2],
        "scope": "post-training evaluation; synthetic test excluded from training",
        "real_validation_caution": "synthetic training images derive from the same 66 backgrounds",
        "real_test_caution": "previously opened competition test; supplementary comparison only",
    }
    (EVAL / "protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False))
    print(json.dumps({"prepared": True, "jobs": len(jobs), "directory": str(EVAL)}))


if __name__ == "__main__":
    main()
