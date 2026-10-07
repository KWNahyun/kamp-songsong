"""Three-seed comparison of MAL, UQ, BR, and base/refined pair selection."""
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
from analyze_pilot import coverage, dump_csv  # noqa: E402


SEEDS = [20260929, 20260930, 20261001]


def paths(seed: int) -> dict[str, Path]:
    return {
        "MAL": ROOT / f"experiments/kamp_ablation_v3/runs/dfine_M_seed{seed}",
        "UQ": ROOT / f"experiments/kamp_relative_selector_v1/runs_frozen/dfine_frozen_UQ_seed{seed}",
        "BR": HERE / f"runs/dfine_frozen_BR_seed{seed}",
        "BPS": HERE / f"runs/dfine_frozen_BPS_seed{seed}",
    }


def detector_identity(seed: int, name: str, run: Path) -> bool:
    checkpoint = torch.load(
        ROOT / f"experiments/kamp_ablation_v3/runs/dfine_M_seed{seed}/best_stg1.pth",
        map_location="cpu",
        weights_only=False,
    )
    control = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    candidate = torch.load(run / "last.pth", map_location="cpu", weights_only=False)["model"]
    return all(key in candidate and torch.equal(value, candidate[key]) for key, value in control.items())


def main() -> None:
    gt = json.loads(
        (ROOT / "data/processed/kamp500_telea_v1/annotations/val.json").read_text()
    )
    ground = {
        image["id"]: [ann for ann in gt["annotations"] if ann["image_id"] == image["id"]]
        for image in gt["images"]
    }
    metric_rows, coverage_rows, identity = [], [], {}
    for seed in SEEDS:
        seed_paths = paths(seed)
        for name in ["UQ", "BR", "BPS"]:
            identity[f"{seed}_{name}"] = detector_identity(seed, name, seed_paths[name])
            if not identity[f"{seed}_{name}"]:
                raise AssertionError(f"Detector changed for {seed} {name}")
        for name, run in seed_paths.items():
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
                        "seed": seed,
                        "model": name,
                        "setting": setting,
                        "AP": metrics["AP"],
                        "AP50": metrics["AP50"],
                        "AP75": metrics["AP75"],
                        "TP_FP6": metrics["operating"]["0.1"]["TP"],
                        "FP": metrics["operating"]["0.1"]["FP"],
                    }
                )
            coverage_summary, _ = coverage(raw, gt)
            coverage_rows.append({"seed": seed, "model": name, **coverage_summary})

    primary = "hard_nms07_scale095"
    paired = []
    for seed in SEEDS:
        base = next(
            row for row in metric_rows
            if row["seed"] == seed and row["model"] == "MAL" and row["setting"] == primary
        )
        for model in ["UQ", "BR", "BPS"]:
            candidate = next(
                row for row in metric_rows
                if row["seed"] == seed and row["model"] == model and row["setting"] == primary
            )
            base_coverage = next(
                row for row in coverage_rows if row["seed"] == seed and row["model"] == "MAL"
            )
            candidate_coverage = next(
                row for row in coverage_rows if row["seed"] == seed and row["model"] == model
            )
            paired.append(
                {
                    "seed": seed,
                    "model": model,
                    **{
                        f"delta_{metric}": candidate[metric] - base[metric]
                        for metric in ["AP", "AP50", "AP75", "TP_FP6"]
                    },
                    "delta_coverage_iou75": (
                        candidate_coverage["coverage_iou75"] - base_coverage["coverage_iou75"]
                    ),
                    "delta_mean_best_iou": (
                        candidate_coverage["mean_best_iou"] - base_coverage["mean_best_iou"]
                    ),
                }
            )

    aggregates = []
    for model in ["UQ", "BR", "BPS"]:
        selected = [row for row in paired if row["model"] == model]
        aggregate = {"model": model}
        for metric in ["AP", "AP50", "AP75", "TP_FP6", "coverage_iou75", "mean_best_iou"]:
            values = [row[f"delta_{metric}"] for row in selected]
            aggregate[f"mean_delta_{metric}"] = float(np.mean(values))
            aggregate[f"std_delta_{metric}"] = float(np.std(values, ddof=1))
            aggregate[f"positive_seeds_{metric}"] = sum(value > 0 for value in values)
            aggregate[f"nonnegative_seeds_{metric}"] = sum(value >= 0 for value in values)
        aggregates.append(aggregate)

    dump_csv(HERE / "confirmation_metrics.csv", metric_rows)
    dump_csv(HERE / "confirmation_coverage.csv", coverage_rows)
    dump_csv(HERE / "confirmation_paired_deltas.csv", paired)
    summary = {
        "split": "validation",
        "seeds": SEEDS,
        "test_used": False,
        "detector_bitwise_identity": identity,
        "primary_setting": primary,
        "paired_deltas_vs_MAL": paired,
        "aggregate_deltas_vs_MAL": aggregates,
        "coverage_at_score_floor_001": coverage_rows,
        "limitations": [
            "Three frozen detector seeds on one temporary preprocessing version.",
            "The MAL base checkpoints and fixed NMS/0.95 settings were previously selected using validation.",
            "The selector/refiner last epochs were evaluated without validation checkpoint selection.",
            "The held-out test split remains untouched.",
        ],
    }
    (HERE / "confirmation_summary.json").write_text(json.dumps(summary, indent=2))

    figure, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    for axis, (metric, title) in zip(
        axes,
        [
            ("AP", "AP50:95"),
            ("AP75", "AP75"),
            ("TP_FP6", "TP at <=6 FP"),
            ("coverage_iou75", "IoU>=.75 candidate coverage"),
        ],
    ):
        for seed in SEEDS:
            if metric == "coverage_iou75":
                source = coverage_rows
            else:
                source = [row for row in metric_rows if row["setting"] == primary]
            values = [
                next(row[metric] for row in source if row["seed"] == seed and row["model"] == model)
                for model in ["MAL", "UQ", "BPS"]
            ]
            axis.plot([0, 1, 2], values, marker="o", label=str(seed))
        axis.set_xticks([0, 1, 2], ["MAL", "UQ", "BPS"])
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(title="seed")
    figure.suptitle("Frozen quality and boundary pair selection: paired three-seed validation")
    figure.tight_layout()
    figure.savefig(HERE / "confirmation_comparison.png", dpi=180)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

