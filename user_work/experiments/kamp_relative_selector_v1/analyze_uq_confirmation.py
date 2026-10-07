"""Three-seed confirmation of the frozen unary quality selector."""
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


SEEDS = [20260929, 20260930, 20261001]


def detector_identity(seed: int) -> bool:
    control_checkpoint = torch.load(
        ROOT / f"experiments/kamp_ablation_v3/runs/dfine_M_seed{seed}/best_stg1.pth",
        map_location="cpu",
        weights_only=False,
    )
    control = (
        control_checkpoint["ema"]["module"]
        if "ema" in control_checkpoint
        else control_checkpoint["model"]
    )
    candidate = torch.load(
        HERE / f"runs_frozen/dfine_frozen_UQ_seed{seed}/last.pth",
        map_location="cpu",
        weights_only=False,
    )["model"]
    return all(key in candidate and torch.equal(value, candidate[key]) for key, value in control.items())


def main() -> None:
    gt = json.loads(
        (ROOT / "data/processed/kamp500_telea_v1/annotations/val.json").read_text()
    )
    ground = {
        image["id"]: [ann for ann in gt["annotations"] if ann["image_id"] == image["id"]]
        for image in gt["images"]
    }
    rows, ranking_rows = [], []
    identities = {}
    for seed in SEEDS:
        identities[str(seed)] = detector_identity(seed)
        if not identities[str(seed)]:
            raise AssertionError(f"Frozen detector changed for seed {seed}")
        runs = {
            "MAL": ROOT / f"experiments/kamp_ablation_v3/runs/dfine_M_seed{seed}",
            "Frozen-UQ": HERE / f"runs_frozen/dfine_frozen_UQ_seed{seed}",
        }
        for model, run in runs.items():
            raw = json.loads((run / "common_eval/predictions_original.json").read_text())
            hard, _ = hard_nms(raw, 0.7)
            settings = {
                "raw_scale095": scale_boxes(raw, 0.95),
                "hard_nms07_scale095": scale_boxes(hard, 0.95),
            }
            for setting, predictions in settings.items():
                metrics = evaluate_setting(gt, ground, predictions)
                rows.append(
                    {
                        "seed": seed,
                        "model": model,
                        "setting": setting,
                        "AP": metrics["AP"],
                        "AP50": metrics["AP50"],
                        "AP75": metrics["AP75"],
                        "TP_FP6": metrics["operating"]["0.1"]["TP"],
                        "FP": metrics["operating"]["0.1"]["FP"],
                        "threshold_FP6": metrics["operating"]["0.1"]["threshold"],
                    }
                )
            ranking_rows.append(
                {
                    "seed": seed,
                    "model": model,
                    **candidate_ranking(settings["raw_scale095"], gt),
                }
            )

    deltas = []
    for seed in SEEDS:
        for setting in ["raw_scale095", "hard_nms07_scale095"]:
            base = next(row for row in rows if row["seed"] == seed and row["model"] == "MAL" and row["setting"] == setting)
            uq = next(row for row in rows if row["seed"] == seed and row["model"] == "Frozen-UQ" and row["setting"] == setting)
            deltas.append(
                {
                    "seed": seed,
                    "setting": setting,
                    **{f"delta_{metric}": uq[metric] - base[metric] for metric in ["AP", "AP50", "AP75", "TP_FP6"]},
                }
            )
    aggregates = []
    for setting in ["raw_scale095", "hard_nms07_scale095"]:
        selected = [row for row in deltas if row["setting"] == setting]
        aggregate = {"setting": setting}
        for metric in ["AP", "AP50", "AP75", "TP_FP6"]:
            values = [row[f"delta_{metric}"] for row in selected]
            aggregate[f"mean_delta_{metric}"] = float(np.mean(values))
            aggregate[f"std_delta_{metric}"] = float(np.std(values, ddof=1))
            aggregate[f"positive_seeds_{metric}"] = sum(value > 0 for value in values)
            aggregate[f"nonnegative_seeds_{metric}"] = sum(value >= 0 for value in values)
        aggregates.append(aggregate)

    dump_csv(HERE / "uq_confirmation_metrics.csv", rows)
    dump_csv(HERE / "uq_confirmation_deltas.csv", deltas)
    dump_csv(HERE / "uq_confirmation_ranking.csv", ranking_rows)
    summary = {
        "split": "validation",
        "seeds": SEEDS,
        "test_used": False,
        "detector_bitwise_identical_by_seed": identities,
        "selector_checkpoint": "last epoch 30; no validation selection",
        "fixed_postprocess": {"box_scale": 0.95, "hard_nms_iou": 0.7},
        "paired_deltas": deltas,
        "aggregate_deltas": aggregates,
        "ranking": ranking_rows,
        "limitations": [
            "All selectors use previously validation-selected MAL base checkpoints.",
            "Only three seeds and one temporary preprocessing version were tested.",
            "The held-out test split remains untouched.",
        ],
    }
    (HERE / "uq_confirmation_summary.json").write_text(json.dumps(summary, indent=2))

    figure, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    setting = "hard_nms07_scale095"
    for axis, (metric, title) in zip(
        axes,
        [("AP", "AP50:95"), ("AP75", "AP75"), ("TP_FP6", "TP at <=6 FP")],
    ):
        for seed in SEEDS:
            base = next(row[metric] for row in rows if row["seed"] == seed and row["model"] == "MAL" and row["setting"] == setting)
            uq = next(row[metric] for row in rows if row["seed"] == seed and row["model"] == "Frozen-UQ" and row["setting"] == setting)
            axis.plot([0, 1], [base, uq], marker="o", label=str(seed))
        axis.set_xticks([0, 1], ["MAL", "Frozen-UQ"])
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(title="seed")
    figure.suptitle("Frozen unary quality selector: paired three-seed validation")
    figure.tight_layout()
    figure.savefig(HERE / "uq_confirmation.png", dpi=180)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

