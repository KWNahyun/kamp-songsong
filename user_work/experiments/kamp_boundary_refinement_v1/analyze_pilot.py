"""Mechanistic and detector analysis for the boundary refinement pilot."""
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


ROOT = Path("/home/viplab/contest")
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT / "experiments/kamp_candidate_diagnostics_v1"))

from analyze_candidates import evaluate_setting, hard_nms, iou, scale_boxes  # noqa: E402


RUNS = {
    "MAL": ROOT / "experiments/kamp_ablation_v3/runs/dfine_M_seed20260929",
    "BR": HERE / "runs/dfine_frozen_BR_seed20260929",
    "EBR": HERE / "runs/dfine_frozen_EBR_seed20260929",
    "BPS": HERE / "runs/dfine_frozen_BPS_seed20260929",
}


def dump_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def coverage(predictions: list[dict], gt: dict) -> tuple[dict, list[dict]]:
    grouped = defaultdict(list)
    for prediction in predictions:
        grouped[prediction["image_id"]].append(prediction)
    rows = []
    for ann in gt["annotations"]:
        best = max(
            (iou(ann["bbox"], prediction["bbox"]) for prediction in grouped[ann["image_id"]]),
            default=0,
        )
        rows.append(
            {
                "gt_id": ann["id"],
                "image_id": ann["image_id"],
                "short_side": min(ann["bbox"][2:]),
                "best_iou": best,
                "iou50": int(best >= 0.5),
                "iou75": int(best >= 0.75),
            }
        )
    summary = {
        "mean_best_iou": float(np.mean([row["best_iou"] for row in rows])),
        "coverage_iou50": sum(row["iou50"] for row in rows),
        "coverage_iou75": sum(row["iou75"] for row in rows),
    }
    return summary, rows


def verify_detector_state(name: str) -> dict:
    checkpoint = torch.load(RUNS["MAL"] / "best_stg1.pth", map_location="cpu", weights_only=False)
    control = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    candidate = torch.load(RUNS[name] / "last.pth", map_location="cpu", weights_only=False)["model"]
    keys = [key for key in control if key in candidate]
    unequal = [key for key in keys if not torch.equal(control[key], candidate[key])]
    return {
        "compared_detector_tensors": len(keys),
        "unequal_detector_tensors": len(unequal),
        "bitwise_identical": not unequal,
    }


def main() -> None:
    gt = json.loads(
        (ROOT / "data/processed/kamp500_telea_v1/annotations/val.json").read_text()
    )
    ground = {
        image["id"]: [ann for ann in gt["annotations"] if ann["image_id"] == image["id"]]
        for image in gt["images"]
    }
    metric_rows, coverage_rows, coverage_by_model = [], [], {}
    per_gt = {}
    for name, run in RUNS.items():
        raw = json.loads((run / "common_eval/predictions_original.json").read_text())
        hard, _ = hard_nms(raw, 0.7)
        settings = {
            "raw": raw,
            "hard_nms07": hard,
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
                }
            )
        summary, detail = coverage(raw, gt)
        coverage_by_model[name] = summary
        per_gt[name] = {row["gt_id"]: row for row in detail}
        coverage_rows.append({"model": name, **summary})

    transition_rows = []
    for name in ["BR", "EBR", "BPS"]:
        improved75 = worsened75 = 0
        iou_changes = []
        for gt_id, base in per_gt["MAL"].items():
            refined = per_gt[name][gt_id]
            improved75 += int(not base["iou75"] and refined["iou75"])
            worsened75 += int(base["iou75"] and not refined["iou75"])
            iou_changes.append(refined["best_iou"] - base["best_iou"])
        transition_rows.append(
            {
                "model": name,
                "gained_iou75_gt": improved75,
                "lost_iou75_gt": worsened75,
                "net_iou75_gt": improved75 - worsened75,
                "mean_best_iou_change": float(np.mean(iou_changes)),
                "median_best_iou_change": float(np.median(iou_changes)),
                "gt_best_iou_improved": sum(change > 1e-8 for change in iou_changes),
                "gt_best_iou_worsened": sum(change < -1e-8 for change in iou_changes),
            }
        )

    oracle_union_iou = {
        gt_id: max(per_gt["MAL"][gt_id]["best_iou"], per_gt["BR"][gt_id]["best_iou"])
        for gt_id in per_gt["MAL"]
    }
    oracle_union = {
        "mean_best_iou": float(np.mean(list(oracle_union_iou.values()))),
        "coverage_iou50": sum(value >= 0.5 for value in oracle_union_iou.values()),
        "coverage_iou75": sum(value >= 0.75 for value in oracle_union_iou.values()),
        "warning": "GT oracle over the union of base and BR boxes; diagnostic ceiling only.",
    }
    identity = {name: verify_detector_state(name) for name in ["BR", "EBR", "BPS"]}
    if not all(item["bitwise_identical"] for item in identity.values()):
        raise AssertionError("Frozen MAL detector changed")
    dump_csv(HERE / "pilot_metrics.csv", metric_rows)
    dump_csv(HERE / "pilot_coverage.csv", coverage_rows)
    dump_csv(HERE / "pilot_transitions.csv", transition_rows)
    summary = {
        "split": "validation",
        "seed": 20260929,
        "test_used": False,
        "base_detector_frozen": True,
        "detector_identity": identity,
        "coverage_at_score_floor_001": coverage_by_model,
        "base_br_union_oracle": oracle_union,
        "iou75_transitions": transition_rows,
        "metrics": metric_rows,
        "fixed_settings": {
            "hard_nms_iou": 0.7,
            "optional_existing_box_scale": 0.95,
            "score_floor": 0.001,
        },
        "limitations": [
            "Single-seed pilot on the temporary Telea preprocessing.",
            "The MAL base checkpoint was previously selected on validation.",
            "No validation-based refiner checkpoint selection was used.",
            "The held-out test split remains untouched.",
        ],
    }
    (HERE / "pilot_summary.json").write_text(json.dumps(summary, indent=2))

    figure, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    names = list(RUNS)
    for axis, (metric, title, setting) in zip(
        axes[:3],
        [
            ("AP", "AP50:95", "hard_nms07"),
            ("AP75", "AP75", "hard_nms07"),
            ("TP_FP6", "TP at <=6 FP", "hard_nms07"),
        ],
    ):
        values = [
            next(row[metric] for row in metric_rows if row["model"] == name and row["setting"] == setting)
            for name in names
        ]
        axis.bar(names, values)
        axis.set_title(title + "; no 0.95 scale")
        axis.grid(axis="y", alpha=0.25)
    axes[3].bar(names, [coverage_by_model[name]["coverage_iou75"] for name in names])
    axes[3].set_title("GT with IoU>=.75 candidate")
    axes[3].grid(axis="y", alpha=0.25)
    figure.suptitle("Frozen boundary refinement pilot; seed 20260929")
    figure.tight_layout()
    figure.savefig(HERE / "pilot_comparison.png", dpi=180)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
