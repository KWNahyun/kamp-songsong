from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Korean labels must render in the Markdown-linked PNG/PDF figures.
plt.rcParams["font.family"] = "Noto Sans CJK KR"
plt.rcParams["axes.unicode_minus"] = False


ROOT = Path("/home/viplab/contest")
HERE = Path(__file__).parent
OLD = ROOT / "experiments/kamp_synth_v1_eval"
OLD_REAL = ROOT / "experiments/kamp_v2_test_frozen"
SUMMARY = HERE / "summary"
SUMMARY.mkdir(exist_ok=True)
VARIANTS = ["yolov8s", "yolov8s_p2", "dfine_s", "dfine_s_p2", "dfine_M", "dfine_mal_UQ"]
SEEDS = [20260929, 20260930, 20261001]


def old_name(variant: str, seed: int) -> str:
    return f"{variant}_seed{seed}"


def new_name(variant: str, seed: int) -> str:
    return f"{variant}_synth25_seed{seed}"


def synthetic_run(path: Path, name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    objects = pd.read_csv(path / "runs" / name / "analysis/objects.csv")
    images = pd.read_csv(path / "runs" / name / "analysis/images.csv")
    return objects, images


def main() -> None:
    rows = []
    conditions = []
    old_test = {row["name"]: row for row in json.loads((OLD_REAL / "summary.json").read_text())["runs"]}
    protocol = json.loads((HERE / "protocol.json").read_text())
    old_protocol = {(row["variant"], row["seed"]): row for row in json.loads((OLD_REAL / "frozen_protocol.json").read_text())["jobs"]}

    for variant in VARIANTS:
        for seed in SEEDS:
            before_name = old_name(variant, seed)
            after_name = new_name(variant, seed)
            before_objects, before_images = synthetic_run(OLD, before_name)
            after_objects, after_images = synthetic_run(HERE, after_name)
            before_test_objects = before_objects[before_objects.split == "test"]
            after_test_objects = after_objects[after_objects.split == "test"]
            before_erased = before_images[(before_images.split == "test") & (before_images["set"] == "erased")]
            after_erased = after_images[(after_images.split == "test") & (after_images["set"] == "erased")]
            real_after = json.loads((HERE / "real" / after_name / "metrics.json").read_text())
            real_before = old_test[before_name]
            val_before = old_protocol[(variant, seed)]
            rows.append({
                "variant": variant, "seed": seed,
                "synth_detection_before": before_test_objects.found.mean(),
                "synth_detection_after": after_test_objects.found.mean(),
                "synth_detection_delta": after_test_objects.found.mean() - before_test_objects.found.mean(),
                "erased_alarm_before": (before_erased["count"] > 0).mean(),
                "erased_alarm_after": (after_erased["count"] > 0).mean(),
                "erased_alarm_delta": (after_erased["count"] > 0).mean() - (before_erased["count"] > 0).mean(),
                "real_val_AP_before": val_before["val_AP"],
                "real_val_AP_after": real_after["val"]["AP"],
                "real_val_AP75_before": val_before["val_AP75"],
                "real_val_AP75_after": real_after["val"]["AP75"],
                "real_test_AP_before": real_before["AP"],
                "real_test_AP_after": real_after["test"]["AP"],
                "real_test_AP75_before": real_before["AP75"],
                "real_test_AP75_after": real_after["test"]["AP75"],
                "real_test_TP_before_locked": real_before["TP"],
                "real_test_TP_after_locked": real_after["test"]["TP"],
                "real_test_FP_before_locked": real_before["FP"],
                "real_test_FP_after_locked": real_after["test"]["FP"],
                "old_locked_threshold": real_after["test"]["locked_threshold"],
                "diagnostic_recalibrated_threshold": real_after["val"]["recalibrated_FP6"]["threshold"],
            })
            for condition in sorted(set(before_test_objects["set"]) | set(after_test_objects["set"])):
                before = before_test_objects[before_test_objects["set"] == condition]
                after = after_test_objects[after_test_objects["set"] == condition]
                if len(before) == 0 or len(after) == 0:
                    continue
                conditions.append({
                    "variant": variant, "seed": seed, "condition": condition,
                    "objects": len(after), "before": before.found.mean(), "after": after.found.mean(),
                    "delta": after.found.mean() - before.found.mean(),
                })

    runs = pd.DataFrame(rows)
    runs.to_csv(SUMMARY / "run_comparison.csv", index=False)
    condition_runs = pd.DataFrame(conditions)
    condition_runs.to_csv(SUMMARY / "condition_run_comparison.csv", index=False)

    numeric = [column for column in runs.columns if column not in ["variant", "seed"]]
    aggregate = runs.groupby("variant")[numeric].agg(["mean", "std"])
    aggregate.columns = [f"{column}_{stat}" for column, stat in aggregate.columns]
    aggregate.reset_index().to_csv(SUMMARY / "model_comparison.csv", index=False)
    condition_aggregate = condition_runs.groupby(["variant", "condition"]).agg(
        objects=("objects", "first"), before_mean=("before", "mean"), after_mean=("after", "mean"),
        delta_mean=("delta", "mean"), delta_sd=("delta", "std")
    ).reset_index()
    condition_aggregate.to_csv(SUMMARY / "condition_comparison.csv", index=False)

    labels = ["YOLO", "YOLO+P2", "D-FINE", "D-FINE+P2", "D-FINE+MAL", "D-FINE+MAL+UQ"]
    model = pd.read_csv(SUMMARY / "model_comparison.csv").set_index("variant").loc[VARIANTS]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.7))
    x = np.arange(len(VARIANTS))
    width = 0.36
    axes[0].bar(x - width / 2, model.synth_detection_before_mean * 100, width, label="학습 전")
    axes[0].bar(x + width / 2, model.synth_detection_after_mean * 100, width, label="합성 보완 학습 후")
    axes[0].set(title="합성 test 전체 이물 주변 경보 도달률", ylabel="도달률 (%)", xticks=x, xticklabels=labels)
    axes[1].bar(x - width / 2, model.real_test_AP_before_mean, width, label="학습 전")
    axes[1].bar(x + width / 2, model.real_test_AP_after_mean, width, label="합성 보완 학습 후")
    axes[1].set(title="실제 test AP", ylabel="AP", xticks=x, xticklabels=labels)
    axes[2].bar(x - width / 2, model.erased_alarm_before_mean * 100, width, label="학습 전")
    axes[2].bar(x + width / 2, model.erased_alarm_after_mean * 100, width, label="합성 보완 학습 후")
    axes[2].set(title="제거 배경 경보 영상 비율", ylabel="경보 영상 (%)", xticks=x, xticklabels=labels)
    for axis in axes:
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.2)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(SUMMARY / "training_before_after.png", dpi=200)
    figure.savefig(SUMMARY / "training_before_after.pdf")
    plt.close(figure)

    focus = ["pos_random_k0.5", "gvxr_size_x0.5", "gvxr_size_x0.75", "gvxr_shape_wire10", "gvxr_shape_wire20"]
    selected = condition_aggregate[(condition_aggregate.variant == "dfine_mal_UQ") & condition_aggregate.condition.isin(focus)].set_index("condition").reindex(focus)
    figure, axis = plt.subplots(figsize=(9, 5))
    xx = np.arange(len(focus))
    axis.bar(xx - width / 2, selected.before_mean * 100, width, label="학습 전")
    axis.bar(xx + width / 2, selected.after_mean * 100, width, label="합성 보완 학습 후")
    axis.set(xticks=xx, xticklabels=["이동·저대비", "크기 0.5배", "크기 0.75배", "선형 10배", "선형 20배"], ylabel="도달률 (%)", title="대표 모델의 취약 조건별 보완 효과")
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(SUMMARY / "final_model_weak_conditions.png", dpi=200)
    figure.savefig(SUMMARY / "final_model_weak_conditions.pdf")
    plt.close(figure)

    (SUMMARY / "complete.json").write_text(json.dumps({"complete": True, "runs": len(runs), "conditions": len(condition_runs)}, indent=2))
    print(model[["synth_detection_before_mean", "synth_detection_after_mean", "real_test_AP_before_mean", "real_test_AP_after_mean", "erased_alarm_before_mean", "erased_alarm_after_mean"]].to_string())


if __name__ == "__main__":
    main()
