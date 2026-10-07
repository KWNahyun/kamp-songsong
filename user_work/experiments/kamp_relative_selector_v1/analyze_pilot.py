"""Analyze MAL, unary quality, and relative query selection on validation."""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr


ROOT = Path("/home/viplab/contest")
HERE = Path(__file__).parent
DIAGNOSTICS = ROOT / "experiments/kamp_candidate_diagnostics_v1"
sys.path.insert(0, str(DIAGNOSTICS))

from analyze_candidates import (  # noqa: E402
    SCORE_FLOOR,
    evaluate_setting,
    hard_nms,
    iou,
    scale_boxes,
)


RUNS = {
    "MAL": ROOT / "experiments/kamp_ablation_v3/runs/dfine_M_seed20260929",
    "UQ": HERE / "runs/dfine_UQ_seed20260929",
    "RQS": HERE / "runs/dfine_RQS_seed20260929",
}


def dump_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def candidate_ranking(predictions: list[dict], gt: dict) -> dict:
    grouped = defaultdict(list)
    for prediction in predictions:
        grouped[prediction["image_id"]].append(prediction)
    states = defaultdict(int)
    candidate_scores, candidate_qualities = [], []
    for image_id, image_predictions in grouped.items():
        truth = [ann for ann in gt["annotations"] if ann["image_id"] == image_id]
        for prediction in image_predictions:
            candidate_scores.append(prediction["score"])
            candidate_qualities.append(
                max((iou(prediction["bbox"], ann["bbox"]) for ann in truth), default=0)
            )
    for ann in gt["annotations"]:
        image_predictions = grouped[ann["image_id"]]
        overlaps = [iou(ann["bbox"], prediction["bbox"]) for prediction in image_predictions]
        best = max(overlaps, default=0)
        near = [index for index, overlap in enumerate(overlaps) if overlap >= 0.1]
        leader = max(near, key=lambda index: image_predictions[index]["score"]) if near else None
        leader_iou = overlaps[leader] if leader is not None else 0
        if best < 0.1:
            state = "no_near_candidate"
        elif best < 0.5:
            state = "near_but_below_iou50"
        elif best < 0.75:
            state = "iou50_but_no_iou75"
        elif leader_iou < 0.75:
            state = "iou75_exists_bad_score_leader"
        else:
            state = "iou75_and_good_leader"
        states[state] += 1
    correlation = spearmanr(candidate_scores, candidate_qualities).statistic
    return {
        **states,
        "candidate_score_iou_spearman": float(correlation),
    }


def compare_last_detector_states() -> dict:
    control = torch.load(RUNS["MAL"] / "last.pth", map_location="cpu", weights_only=False)["model"]
    result = {}
    for name in ["UQ", "RQS"]:
        candidate = torch.load(RUNS[name] / "last.pth", map_location="cpu", weights_only=False)["model"]
        keys = [key for key in control if key in candidate and "kamp_selector" not in key]
        unequal = [key for key in keys if not torch.equal(control[key], candidate[key])]
        max_difference = max(
            (
                float((control[key].float() - candidate[key].float()).abs().max())
                for key in unequal
            ),
            default=0.0,
        )
        result[name] = {
            "compared_tensors": len(keys),
            "unequal_tensors": len(unequal),
            "max_absolute_difference": max_difference,
            "bitwise_identical": not unequal,
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

    dump_csv(HERE / "pilot_metrics.csv", metric_rows)
    dump_csv(HERE / "candidate_ranking.csv", ranking_rows)
    detector_identity = compare_last_detector_states()
    summary = {
        "split": "validation",
        "seed": 20260929,
        "test_used": False,
        "fixed_postprocess": {"box_scale": 0.95, "hard_nms_iou": 0.7},
        "metrics": metric_rows,
        "candidate_ranking": ranking_rows,
        "last_epoch_detector_identity": detector_identity,
        "limitations": [
            "Single-seed 30-epoch pilot.",
            "Each best checkpoint was selected on the same validation split reported here.",
            "Telea preprocessing is temporary.",
        ],
    }
    (HERE / "pilot_summary.json").write_text(json.dumps(summary, indent=2))

    names = ["MAL", "UQ", "RQS"]
    metrics_to_plot = [("AP", "AP50:95"), ("AP75", "AP75"), ("TP_FP6", "TP at <=6 FP")]
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for axis, (metric, title) in zip(axes, metrics_to_plot):
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
    figure.suptitle("Detached selector pilot; validation seed 20260929")
    figure.tight_layout()
    figure.savefig(HERE / "pilot_comparison.png", dpi=180)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

