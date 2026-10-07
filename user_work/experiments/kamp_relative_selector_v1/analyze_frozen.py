"""Clean frozen-base analysis for unary and relative candidate selectors."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path("/home/viplab/contest")
HERE = Path(__file__).parent
sys.path[:0] = [str(HERE), str(ROOT / "experiments/kamp_candidate_diagnostics_v1")]

from analyze_candidates import evaluate_setting, hard_nms, scale_boxes  # noqa: E402
from analyze_pilot import candidate_ranking, dump_csv  # noqa: E402


RUNS = {
    "MAL": ROOT / "experiments/kamp_ablation_v3/runs/dfine_M_seed20260929",
    "Frozen-UQ": HERE / "runs_frozen/dfine_frozen_UQ_seed20260929",
    "Frozen-RQS": HERE / "runs_frozen/dfine_frozen_RQS_seed20260929",
}


def verify_frozen_state() -> dict:
    checkpoint = torch.load(
        RUNS["MAL"] / "best_stg1.pth", map_location="cpu", weights_only=False
    )
    control = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    result = {}
    for name in ["Frozen-UQ", "Frozen-RQS"]:
        candidate = torch.load(
            RUNS[name] / "last.pth", map_location="cpu", weights_only=False
        )["model"]
        keys = [key for key in control if key in candidate]
        unequal = [key for key in keys if not torch.equal(control[key], candidate[key])]
        result[name] = {
            "compared_detector_tensors": len(keys),
            "unequal_detector_tensors": len(unequal),
            "bitwise_identical_to_control": not unequal,
        }
    return result


def main() -> None:
    gt = json.loads(
        (ROOT / "data/processed/kamp500_telea_v1/annotations/val.json").read_text()
    )
    ground = {
        image["id"]: [ann for ann in gt["annotations"] if ann["image_id"] == image["id"]]
        for image in gt["images"]
    }
    metric_rows, ranking_rows = [], []
    for name, run in RUNS.items():
        raw = json.loads((run / "common_eval/predictions_original.json").read_text())
        hard, _ = hard_nms(raw, 0.7)
        settings = {
            "raw_scale095": scale_boxes(raw, 0.95),
            "hard_nms07_scale095": scale_boxes(hard, 0.95),
        }
        for setting, predictions in settings.items():
            metrics = evaluate_setting(gt, ground, predictions)
            metric_rows.append(
                {
                    "model": name,
                    "setting": setting,
                    "AP": metrics["AP"],
                    "AP50": metrics["AP50"],
                    "AP75": metrics["AP75"],
                    "TP_FP6": metrics["operating"]["0.1"]["TP"],
                    "FP": metrics["operating"]["0.1"]["FP"],
                    "threshold_FP6": metrics["operating"]["0.1"]["threshold"],
                }
            )
        ranking_rows.append({"model": name, **candidate_ranking(settings["raw_scale095"], gt)})

    identity = verify_frozen_state()
    if not all(item["bitwise_identical_to_control"] for item in identity.values()):
        raise AssertionError("Frozen detector changed")
    dump_csv(HERE / "frozen_pilot_metrics.csv", metric_rows)
    dump_csv(HERE / "frozen_candidate_ranking.csv", ranking_rows)
    summary = {
        "split": "validation",
        "seed": 20260929,
        "test_used": False,
        "base_detector": "MAL seed20260929 best_stg1 EMA",
        "base_frozen": True,
        "selector_checkpoint": "last epoch 30; validation not used for selector training or selection",
        "fixed_postprocess": {"box_scale": 0.95, "hard_nms_iou": 0.7},
        "detector_identity": identity,
        "metrics": metric_rows,
        "candidate_ranking": ranking_rows,
        "limitations": [
            "Single frozen detector seed.",
            "The frozen MAL base checkpoint was previously selected on validation.",
            "Telea preprocessing is temporary.",
        ],
    }
    (HERE / "frozen_summary.json").write_text(json.dumps(summary, indent=2))

    names = list(RUNS)
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for axis, (metric, title) in zip(
        axes,
        [("AP", "AP50:95"), ("AP75", "AP75"), ("TP_FP6", "TP at <=6 FP")],
    ):
        x = np.arange(len(names))
        raw_values = [
            next(row[metric] for row in metric_rows if row["model"] == name and row["setting"] == "raw_scale095")
            for name in names
        ]
        nms_values = [
            next(row[metric] for row in metric_rows if row["model"] == name and row["setting"] == "hard_nms07_scale095")
            for name in names
        ]
        axis.bar(x - 0.18, raw_values, width=0.36, label="raw + scale")
        axis.bar(x + 0.18, nms_values, width=0.36, label="NMS + scale")
        axis.set_xticks(x, names)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend()
    figure.suptitle("Frozen MAL candidate selector pilot; validation only")
    figure.tight_layout()
    figure.savefig(HERE / "frozen_pilot_comparison.png", dpi=180)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

