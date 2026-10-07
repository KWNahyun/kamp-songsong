"""Frozen-candidate diagnostics for the three MAL validation predictions.

This script never reads the test split and never trains a model.  The two
oracle settings use ground truth and are diagnostic ceilings, not inference
methods.
"""
from __future__ import annotations

import contextlib
import copy
import csv
import hashlib
import io
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from scipy.optimize import linear_sum_assignment
from torchvision.ops import nms as torchvision_nms


ROOT = Path("/home/viplab/contest")
OUT = ROOT / "experiments/kamp_candidate_diagnostics_v1"
PILOT = ROOT / "experiments/kamp_pilot_v1"
V3 = ROOT / "experiments/kamp_ablation_v3"
DATA = ROOT / "data/processed/kamp500_telea_v1"
SEEDS = [20260929, 20260930, 20261001]
RUNS = {s: V3 / "runs" / f"dfine_M_seed{s}" for s in SEEDS}
SCORE_FLOOR = 0.001
FIXED_SCALE = 0.95
HARD_NMS_IOU = 0.7
SOFT_LINEAR_IOU = 0.7
SOFT_GAUSSIAN_SIGMA = 0.5

sys.path.insert(0, str(PILOT))
from analyze_failures import iou, match, operating  # noqa: E402


def dump_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def scale_boxes(predictions: list[dict], factor: float = FIXED_SCALE) -> list[dict]:
    result = copy.deepcopy(predictions)
    for pred in result:
        x, y, w, h = pred["bbox"]
        pred["bbox"] = [x + (1 - factor) * w / 2, y + (1 - factor) * h / 2, w * factor, h * factor]
    return result


def coco_metrics(gt: dict, predictions: list[dict]) -> tuple[float, float, float]:
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO()
        coco.dataset = gt
        coco.createIndex()
        if not predictions:
            return 0.0, 0.0, 0.0
        ev = COCOeval(coco, coco.loadRes(predictions), "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return tuple(100 * float(x) for x in ev.stats[:3])


def hard_nms(predictions: list[dict], threshold: float) -> tuple[list[dict], list[dict]]:
    output, changes = [], []
    by_image = defaultdict(list)
    for index, pred in enumerate(predictions):
        by_image[pred["image_id"]].append((index, copy.deepcopy(pred)))
    for image_id, indexed in by_image.items():
        boxes = torch.tensor([pred["bbox"] for _, pred in indexed], dtype=torch.float32)
        boxes[:, 2:] += boxes[:, :2]
        scores = torch.tensor([pred["score"] for _, pred in indexed], dtype=torch.float32)
        kept_local = torchvision_nms(boxes, scores, threshold).tolist()
        kept = [indexed[k] for k in kept_local]
        output.extend(pred for _, pred in kept)
        kept_set = set(kept_local)
        for local_index, (idx, pred) in enumerate(indexed):
            if local_index in kept_set:
                continue
            suppressors = [
                (keep_idx, keep, iou(keep["bbox"], pred["bbox"]))
                for keep_idx, keep in kept
                if keep["score"] >= pred["score"] and iou(keep["bbox"], pred["bbox"]) > threshold
            ]
            keep_idx, keep, overlap = max(suppressors, key=lambda x: x[1]["score"])
            changes.append({
                "image_id": image_id,
                "candidate_index": idx,
                "action": "deleted",
                "old_score": pred["score"],
                "new_score": 0.0,
                "suppressor_index": keep_idx,
                "suppressor_score": keep["score"],
                "pair_iou": overlap,
            })
    return output, changes


def soft_nms(
    predictions: list[dict], method: str, threshold: float = SOFT_LINEAR_IOU,
    sigma: float = SOFT_GAUSSIAN_SIGMA, score_floor: float = SCORE_FLOOR,
) -> tuple[list[dict], list[dict]]:
    """Canonical greedy Soft-NMS with a fixed final score floor."""
    output, changes = [], []
    by_image = defaultdict(list)
    for index, pred in enumerate(predictions):
        by_image[pred["image_id"]].append((index, copy.deepcopy(pred)))
    for image_id, indexed in by_image.items():
        remaining = indexed
        while remaining:
            best_at = max(range(len(remaining)), key=lambda k: remaining[k][1]["score"])
            best_idx, best = remaining.pop(best_at)
            if best["score"] < score_floor:
                break
            output.append(best)
            updated = []
            for idx, pred in remaining:
                overlap = iou(best["bbox"], pred["bbox"])
                old = pred["score"]
                if method == "linear":
                    weight = 1 - overlap if overlap > threshold else 1.0
                elif method == "gaussian":
                    weight = math.exp(-(overlap * overlap) / sigma)
                else:
                    raise ValueError(method)
                pred["score"] = old * weight
                if pred["score"] != old:
                    changes.append({
                        "image_id": image_id,
                        "candidate_index": idx,
                        "action": "decayed" if pred["score"] >= score_floor else "fell_below_floor",
                        "old_score": old,
                        "new_score": pred["score"],
                        "suppressor_index": best_idx,
                        "suppressor_score": best["score"],
                        "pair_iou": overlap,
                    })
                if pred["score"] >= score_floor:
                    updated.append((idx, pred))
            remaining = updated
    return output, changes


def evaluate_setting(gt: dict, ground: dict[int, list[dict]], predictions: list[dict]) -> dict:
    ap, ap50, ap75 = coco_metrics(gt, predictions)
    events = match(predictions, ground)
    budgets = {}
    for budget in [0.05, 0.1, 0.2, 0.5]:
        op = operating(events, budget, len(gt["images"]))
        budgets[str(budget)] = {"TP": op["TP"], "FP": op["FP"], "threshold": op["threshold"]}
    return {
        "AP": ap, "AP50": ap50, "AP75": ap75,
        "predictions": len(predictions), "operating": budgets,
    }


def oracle_rerank(predictions: list[dict], ground: dict[int, list[dict]]) -> list[dict]:
    result = []
    for pred in predictions:
        quality = max((iou(pred["bbox"], ann["bbox"]) for ann in ground[pred["image_id"]]), default=0.0)
        if quality >= SCORE_FLOOR:
            item = copy.deepcopy(pred)
            item["score"] = quality
            result.append(item)
    return result


def oracle_unique_best(predictions: list[dict], ground: dict[int, list[dict]]) -> list[dict]:
    """One-to-one maximum-IoU assignment per image, scored by IoU."""
    result = []
    by_image = defaultdict(list)
    for pred in predictions:
        by_image[pred["image_id"]].append(pred)
    for image_id, anns in ground.items():
        preds = by_image[image_id]
        if not anns or not preds:
            continue
        matrix = np.array([[iou(ann["bbox"], pred["bbox"]) for pred in preds] for ann in anns])
        ann_ids, pred_ids = linear_sum_assignment(-matrix)
        for ai, pi in zip(ann_ids, pred_ids):
            quality = float(matrix[ai, pi])
            if quality < SCORE_FLOOR:
                continue
            item = copy.deepcopy(preds[pi])
            item["score"] = quality
            result.append(item)
    return result


def candidate_rows(
    seed: int, raw_scaled: list[dict], methods: dict[str, list[dict]],
    gt: dict, ground: dict[int, list[dict]], descriptors: dict[int, dict], metrics: dict,
) -> list[dict]:
    rows = []
    raw_by_image = defaultdict(list)
    for pred in raw_scaled:
        raw_by_image[pred["image_id"]].append(pred)
    method_by_image = {}
    for method, predictions in methods.items():
        grouped = defaultdict(list)
        for pred in predictions:
            grouped[pred["image_id"]].append(pred)
        method_by_image[method] = grouped
    for ann in gt["annotations"]:
        image_id, gt_id = ann["image_id"], ann["id"]
        raw = raw_by_image[image_id]
        raw_overlaps = [iou(ann["bbox"], pred["bbox"]) for pred in raw]
        best_index = int(np.argmax(raw_overlaps)) if raw_overlaps else -1
        best_iou = raw_overlaps[best_index] if best_index >= 0 else 0.0
        near = [i for i, overlap in enumerate(raw_overlaps) if overlap >= 0.1]
        leader_index = max(near, key=lambda i: raw[i]["score"]) if near else -1
        best75_scores = [raw[i]["score"] for i, overlap in enumerate(raw_overlaps) if overlap >= 0.75]
        base = {
            "seed": seed, "gt_id": gt_id, "image_id": image_id,
            **descriptors[gt_id],
            "raw_best_iou": best_iou,
            "raw_best_score": raw[best_index]["score"] if best_index >= 0 else None,
            "raw_leader_iou": raw_overlaps[leader_index] if leader_index >= 0 else 0.0,
            "raw_leader_score": raw[leader_index]["score"] if leader_index >= 0 else None,
            "raw_best75_max_score": max(best75_scores) if best75_scores else None,
            "raw_precise_candidate": int(best_iou >= 0.75),
            "raw_recoverable_ranking75": int(best_iou >= 0.75 and (leader_index < 0 or raw_overlaps[leader_index] < 0.75)),
        }
        for method, grouped in method_by_image.items():
            predictions = grouped[image_id]
            overlaps = [iou(ann["bbox"], pred["bbox"]) for pred in predictions]
            op_threshold = metrics[method]["operating"]["0.1"]["threshold"]
            base[f"{method}_best_iou_floor001"] = max(overlaps, default=0.0)
            base[f"{method}_best_iou_score05"] = max(
                [overlap for overlap, pred in zip(overlaps, predictions) if pred["score"] >= 0.05], default=0.0
            )
            base[f"{method}_best_iou_operating"] = max(
                [overlap for overlap, pred in zip(overlaps, predictions) if pred["score"] >= op_threshold], default=0.0
            )
        rows.append(base)
    return rows


def suppression_audit(
    seed: int, changes: dict[str, list[dict]], raw_scaled: list[dict],
    ground: dict[int, list[dict]], image_map: dict[int, dict], descriptors: dict[int, dict],
) -> list[dict]:
    rows = []
    for method, events in changes.items():
        for event in events:
            candidate = raw_scaled[event["candidate_index"]]
            suppressor = raw_scaled[event["suppressor_index"]]
            anns = ground[event["image_id"]]
            c_ious = [iou(candidate["bbox"], ann["bbox"]) for ann in anns]
            s_ious = [iou(suppressor["bbox"], ann["bbox"]) for ann in anns]
            ci = int(np.argmax(c_ious)) if c_ious else -1
            si = int(np.argmax(s_ious)) if s_ious else -1
            c_gt = anns[ci]["id"] if ci >= 0 and c_ious[ci] >= 0.1 else None
            s_gt = anns[si]["id"] if si >= 0 and s_ious[si] >= 0.1 else None
            row = {
                "seed": seed, "method": method, **event,
                "stem": Path(image_map[event["image_id"]]["file_name"]).stem,
                "candidate_best_gt": c_gt,
                "candidate_best_gt_iou": c_ious[ci] if ci >= 0 else 0.0,
                "suppressor_best_gt": s_gt,
                "suppressor_best_gt_iou": s_ious[si] if si >= 0 else 0.0,
                "same_best_gt": int(c_gt is not None and c_gt == s_gt),
                "candidate_precise75": int(ci >= 0 and c_ious[ci] >= 0.75),
                "suppressor_precise75_same_gt": int(c_gt is not None and c_gt == s_gt and s_ious[si] >= 0.75),
                "different_labeled_objects": int(c_gt is not None and s_gt is not None and c_gt != s_gt),
            }
            if c_gt is not None:
                row.update({f"candidate_{k}": v for k, v in descriptors[c_gt].items() if k not in {"gt_id", "image_id"}})
            rows.append(row)
    return rows


def draw_box(draw: ImageDraw.ImageDraw, box: list[float], color: str, width: int = 2) -> None:
    x, y, w, h = box
    draw.rectangle((x, y, x + w, y + h), outline=color, width=width)


def make_case_panels(
    gt: dict, ground: dict[int, list[dict]], image_map: dict[int, dict],
    manifest_by_stem: dict[str, dict], methods_by_seed: dict[int, dict[str, list[dict]]],
    metrics_by_seed: dict[int, dict], repeated: list[dict],
) -> list[dict]:
    case_dir = OUT / "cases"
    case_dir.mkdir(exist_ok=True)
    records = []
    for case in repeated:
        gt_id = int(case["gt_id"])
        ann = next(a for a in gt["annotations"] if a["id"] == gt_id)
        image_id = ann["image_id"]
        info = image_map[image_id]
        stem = Path(info["file_name"]).stem
        manifest = manifest_by_stem[stem]
        processed = Image.open(DATA / "images/val" / info["file_name"]).convert("RGB")
        raw = Image.open(ROOT / "data/raw" / manifest["path"]).convert("RGB")
        gx, gy, gw, gh = ann["bbox"]
        margin = max(35, int(max(gw, gh) * 5))
        crop = (
            max(0, int(gx - margin)), max(0, int(gy - margin)),
            min(info["width"], int(gx + gw + margin)), min(info["height"], int(gy + gh + margin)),
        )
        panels = []
        for label, image in [("raw", raw), ("processed", processed)]:
            view = image.copy()
            d = ImageDraw.Draw(view)
            draw_box(d, ann["bbox"], "#00ff66", 2)
            panels.append((label, view.crop(crop)))
        for seed in SEEDS:
            predictions = [p for p in methods_by_seed[seed]["hard_nms07_scale095"] if p["image_id"] == image_id]
            threshold = metrics_by_seed[seed]["hard_nms07_scale095"]["operating"]["0.1"]["threshold"]
            view = processed.copy()
            d = ImageDraw.Draw(view)
            draw_box(d, ann["bbox"], "#00ff66", 2)
            for pred in predictions:
                if pred["score"] >= threshold:
                    draw_box(d, pred["bbox"], "#ff3b30", 2)
            panels.append((f"MAL {seed}", view.crop(crop)))
        size = (300, 300)
        canvas = Image.new("RGB", (len(panels) * size[0], size[1] + 48), "white")
        dc = ImageDraw.Draw(canvas)
        for index, (label, panel) in enumerate(panels):
            panel.thumbnail((size[0] - 8, size[1] - 8), Image.Resampling.NEAREST)
            x = index * size[0] + (size[0] - panel.width) // 2
            y = 44 + (size[1] - panel.height) // 2
            canvas.paste(panel, (x, y))
            dc.text((index * size[0] + 8, 8), label, fill="black")
        dc.text((8, 25), f"GT {gt_id} {stem}; green=GT, red=prediction at FP<=6 operating point", fill="black")
        filename = f"gt_{gt_id}_{stem}.png"
        canvas.save(case_dir / filename)
        records.append({"gt_id": gt_id, "image_id": image_id, "stem": stem, "reason": case["reason"], "file": filename})
    return records


def main() -> None:
    OUT.mkdir(exist_ok=True)
    gt = json.loads((DATA / "annotations/val.json").read_text())
    image_map = {im["id"]: im for im in gt["images"]}
    ground = {image_id: [a for a in gt["annotations"] if a["image_id"] == image_id] for image_id in image_map}
    descriptors = {
        int(row["gt_id"]): row
        for row in csv.DictReader((PILOT / "failure_analysis/gt_conditions.csv").open(encoding="utf-8-sig"))
    }
    manifest_by_stem = {row["stem"]: row for row in json.loads((DATA / "manifest.json").read_text())}

    metric_rows, candidate_detail, suppression_detail = [], [], []
    metrics_by_seed, methods_by_seed, oracle_results = {}, {}, {}
    for seed, run in RUNS.items():
        raw = json.loads((run / "common_eval/predictions_original.json").read_text())
        scaled = scale_boxes(raw)
        # Preserve the previously registered order: suppression first, then
        # the fixed centre-preserving 0.95 box scaling.
        hard_raw, hard_changes = hard_nms(raw, HARD_NMS_IOU)
        linear_raw, linear_changes = soft_nms(raw, "linear")
        gaussian_raw, gaussian_changes = soft_nms(raw, "gaussian")
        hard = scale_boxes(hard_raw)
        linear = scale_boxes(linear_raw)
        gaussian = scale_boxes(gaussian_raw)
        methods = {
            "raw_scale095": scaled,
            "hard_nms07_scale095": hard,
            "soft_linear07_scale095": linear,
            "soft_gaussian05_scale095": gaussian,
        }
        metrics = {name: evaluate_setting(gt, ground, predictions) for name, predictions in methods.items()}
        metrics_by_seed[seed] = metrics
        methods_by_seed[seed] = methods
        for method, values in metrics.items():
            metric_rows.append({
                "seed": seed, "method": method,
                "AP": values["AP"], "AP50": values["AP50"], "AP75": values["AP75"],
                "predictions": values["predictions"],
                **{f"TP_FP{budget}": op["TP"] for budget, op in values["operating"].items()},
                **{f"FP_FP{budget}": op["FP"] for budget, op in values["operating"].items()},
                "threshold_FP0.1": values["operating"]["0.1"]["threshold"],
            })
        reranked = oracle_rerank(scaled, ground)
        reranked_hard, _ = hard_nms(reranked, HARD_NMS_IOU)
        unique = oracle_unique_best(scaled, ground)
        oracle_results[str(seed)] = {
            "quality_rerank_all": evaluate_setting(gt, ground, reranked),
            "quality_rerank_hard_nms07": evaluate_setting(gt, ground, reranked_hard),
            "unique_best_per_gt": evaluate_setting(gt, ground, unique),
            "warning": "Uses GT at inference-equivalent time; diagnostic ceiling only.",
        }
        candidate_detail.extend(candidate_rows(seed, scaled, methods, gt, ground, descriptors, metrics))
        suppression_detail.extend(suppression_audit(
            seed,
            {"hard_nms07": hard_changes, "soft_linear07": linear_changes, "soft_gaussian05": gaussian_changes},
            scaled, ground, image_map, descriptors,
        ))

    dump_csv(OUT / "postprocess_metrics.csv", metric_rows)
    dump_csv(OUT / "candidate_by_gt.csv", candidate_detail)
    dump_csv(OUT / "suppression_events.csv", suppression_detail)
    (OUT / "oracle_diagnostics.json").write_text(json.dumps(oracle_results, indent=2))

    # Aggregate mutually exclusive candidate states and method preservation.
    state_rows = []
    for seed in SEEDS:
        rows = [r for r in candidate_detail if r["seed"] == seed]
        for row in rows:
            best, leader = float(row["raw_best_iou"]), float(row["raw_leader_iou"])
            state = (
                "no_near_candidate" if best < 0.1 else
                "near_but_below_iou50" if best < 0.5 else
                "iou50_but_no_iou75" if best < 0.75 else
                "iou75_exists_bad_score_leader" if leader < 0.75 else
                "iou75_and_good_leader"
            )
            state_rows.append({"seed": seed, "gt_id": row["gt_id"], "state": state})
    dump_csv(OUT / "candidate_states.csv", state_rows)

    repeated = []
    for gt_id in sorted({int(r["gt_id"]) for r in candidate_detail}):
        rows = [r for r in candidate_detail if int(r["gt_id"]) == gt_id]
        hard_missed = all(float(r["hard_nms07_scale095_best_iou_operating"]) < 0.5 for r in rows)
        raw_no75 = all(float(r["raw_best_iou"]) < 0.75 for r in rows)
        if hard_missed or raw_no75:
            repeated.append({
                "gt_id": gt_id,
                "reason": ";".join(x for x, ok in [
                    ("hard_nms_operating_miss_all_seeds", hard_missed),
                    ("no_raw_iou75_candidate_all_seeds", raw_no75),
                ] if ok),
            })
    dump_csv(OUT / "repeated_failures.csv", repeated)
    case_records = make_case_panels(
        gt, ground, image_map, manifest_by_stem, methods_by_seed, metrics_by_seed, repeated
    )
    dump_csv(OUT / "cases/index.csv", case_records)

    # Compact comparison plot.
    order = ["raw_scale095", "hard_nms07_scale095", "soft_linear07_scale095", "soft_gaussian05_scale095"]
    names = ["Raw", "Hard NMS", "Linear Soft-NMS", "Gaussian Soft-NMS"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for axis, metric, title in zip(axes, ["AP", "AP75", "TP_FP0.1"], ["AP50:95", "AP75", "TP at <=6 FP"]):
        for index, method in enumerate(order):
            values = [float(r[metric]) for r in metric_rows if r["method"] == method]
            axis.scatter([index - .08, index, index + .08], values, s=30)
            axis.plot([index - .2, index + .2], [np.mean(values)] * 2, color="black")
        axis.set_xticks(range(len(names)), names, rotation=25, ha="right")
        axis.set_title(title)
        axis.grid(axis="y", alpha=.25)
    fig.suptitle("MAL frozen candidates; fixed 0.95 box scaling; validation only")
    fig.tight_layout()
    fig.savefig(OUT / "suppression_comparison.png", dpi=180)
    plt.close(fig)

    manifest = {
        "created": "2026-10-01",
        "test_used": False,
        "training_performed": False,
        "input_runs": {str(k): str(v) for k, v in RUNS.items()},
        "fixed_settings": {
            "input_score_floor": SCORE_FLOOR,
            "box_scale": FIXED_SCALE,
            "hard_nms_iou": HARD_NMS_IOU,
            "linear_soft_nms_iou": SOFT_LINEAR_IOU,
            "gaussian_soft_nms_sigma": SOFT_GAUSSIAN_SIGMA,
        },
        "selection_warning": "Settings were fixed before inspecting these outputs; no sweep was performed.",
        "oracle_warning": "Oracle outputs use GT and are diagnostic ceilings, never deployable inference.",
        "source_hashes": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [Path(__file__), DATA / "annotations/val.json", PILOT / "failure_analysis/gt_conditions.csv"]
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"metrics": metric_rows, "repeated": repeated, "cases": case_records}, indent=2))


if __name__ == "__main__":
    main()
