from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml


ROOT = Path("/home/viplab/contest")
HERE = Path(__file__).parent
REAL = ROOT / "experiments/kamp_v2_baselines/data640"
SYNTH = ROOT / "data/synthetic/kamp_synth_test_v1"
DATA = HERE / "data640_mix25"
SEEDS = [20260929, 20260930, 20261001]

# 120/470 = 25.53% synthetic images. These conditions were fixed before training.
QUOTAS = {
    "erased": 24,
    "pos_random_k0.5": 20,
    "gvxr_size_x0.5": 16,
    "gvxr_size_x0.75": 12,
    "gvxr_shape_wire10": 12,
    "gvxr_shape_wire20": 12,
    "gvxr_material_Al_x2": 6,
    "gvxr_material_bone_x2": 6,
    "gvxr_material_glass_x2": 6,
    "gvxr_material_stone_x2": 6,
}


def evenly_by_source(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    """Select deterministically while spreading samples across source backgrounds."""
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in sorted(rows, key=lambda x: (x["source_image"], x["image_id"])):
        groups[row["source_image"]].append(row)
    selected: list[dict[str, str]] = []
    sources = sorted(groups)
    depth = 0
    while len(selected) < count:
        progressed = False
        for source in sources:
            if depth < len(groups[source]):
                selected.append(groups[source][depth])
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            break
        depth += 1
    if len(selected) != count:
        raise RuntimeError(f"Could select only {len(selected)} of {count}")
    return selected


def letterbox(image: np.ndarray) -> tuple[np.ndarray, float, float, int, int]:
    height, width = image.shape[:2]
    new_width = round(width * 640 / max(width, height))
    new_height = round(height * 640 / max(width, height))
    pad_x = (640 - new_width) // 2
    pad_y = (640 - new_height) // 2
    canvas = np.full((640, 640, 3), 114, np.uint8)
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    canvas[pad_y : pad_y + new_height, pad_x : pad_x + new_width] = resized
    return canvas, new_width / width, new_height / height, pad_x, pad_y


def convert_synth_label(source: Path, width: int, height: int, sx: float, sy: float, px: int, py: int) -> list[str]:
    labels: list[str] = []
    if not source.exists():
        return labels
    for line in source.read_text().splitlines():
        if not line.strip():
            continue
        cls, cx, cy, box_w, box_h = map(float, line.split())
        if cls != 0:
            raise ValueError(f"Unexpected class {cls} in {source}")
        cx_px = cx * width * sx + px
        cy_px = cy * height * sy + py
        bw_px = box_w * width * sx
        bh_px = box_h * height * sy
        labels.append(f"0 {cx_px/640:.10f} {cy_px/640:.10f} {bw_px/640:.10f} {bh_px/640:.10f}")
    return labels


def coco_from_yolo(image_dir: Path, label_dir: Path, destination: Path) -> dict[str, int]:
    coco = {"images": [], "annotations": [], "categories": [{"id": 0, "name": "defect"}]}
    annotation_id = 0
    for image_id, image_path in enumerate(sorted(image_dir.glob("*.png")), start=1):
        coco["images"].append({"id": image_id, "file_name": image_path.name, "width": 640, "height": 640})
        label_path = label_dir / f"{image_path.stem}.txt"
        for line in label_path.read_text().splitlines():
            cls, cx, cy, box_w, box_h = map(float, line.split())
            width = box_w * 640
            height = box_h * 640
            x = cx * 640 - width / 2
            y = cy * 640 - height / 2
            annotation_id += 1
            coco["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": int(cls),
                "bbox": [x, y, width, height],
                "area": width * height,
                "iscrowd": 0,
            })
    destination.write_text(json.dumps(coco))
    return {"images": len(coco["images"]), "boxes": len(coco["annotations"])}


def symlink_file(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    destination.symlink_to(source)


def main() -> None:
    for split in ["train", "val"]:
        (DATA / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATA / "labels" / split).mkdir(parents=True, exist_ok=True)
    (DATA / "annotations").mkdir(parents=True, exist_ok=True)
    (HERE / "configs").mkdir(parents=True, exist_ok=True)
    (HERE / "runs").mkdir(parents=True, exist_ok=True)
    (HERE / "logs").mkdir(parents=True, exist_ok=True)

    # Real train and validation remain byte-identical through symlinks.
    for split in ["train", "val"]:
        for image_path in sorted((REAL / "images" / split).glob("*.png")):
            symlink_file(image_path, DATA / "images" / split / f"real__{image_path.name}")
            symlink_file(REAL / "labels" / split / f"{image_path.stem}.txt", DATA / "labels" / split / f"real__{image_path.stem}.txt")

    rows = list(csv.DictReader((SYNTH / "images.csv").open()))
    selected: list[dict[str, str]] = []
    for condition, count in QUOTAS.items():
        candidates = [row for row in rows if row["split"] == "val" and row["set"] == condition]
        selected.extend(evenly_by_source(candidates, count))

    if len(selected) != sum(QUOTAS.values()) or len({row["image_id"] for row in selected}) != len(selected):
        raise RuntimeError("Synthetic selection count or uniqueness failed")

    selection_rows = []
    for row in selected:
        source_image = SYNTH / "images" / "val" / f"{row['image_id']}.png"
        image = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(source_image)
        height, width = image.shape[:2]
        if (width, height) != (int(row["width"]), int(row["height"])):
            raise RuntimeError(f"Size mismatch for {source_image}")
        canvas, sx, sy, px, py = letterbox(image)
        stem = f"synth__{row['image_id']}"
        output_image = DATA / "images" / "train" / f"{stem}.png"
        output_label = DATA / "labels" / "train" / f"{stem}.txt"
        if not cv2.imwrite(str(output_image), canvas):
            raise RuntimeError(f"Could not write {output_image}")
        labels = convert_synth_label(SYNTH / "labels" / "val" / f"{row['image_id']}.txt", width, height, sx, sy, px, py)
        output_label.write_text(("\n".join(labels) + "\n") if labels else "")
        selection_rows.append({**row, "output_image": str(output_image), "boxes": len(labels)})

    train_stats = coco_from_yolo(DATA / "images" / "train", DATA / "labels" / "train", DATA / "annotations" / "train.json")
    val_stats = coco_from_yolo(DATA / "images" / "val", DATA / "labels" / "val", DATA / "annotations" / "val.json")
    (DATA / "data.yaml").write_text(yaml.safe_dump({
        "path": str(DATA), "train": "images/train", "val": "images/val", "names": {0: "defect"}
    }, sort_keys=False))

    with (HERE / "synthetic_selection.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selection_rows[0]))
        writer.writeheader()
        writer.writerows(selection_rows)

    # D-FINE configs: fine-tune existing real-data checkpoints at a lower LR.
    jobs = []
    for seed in SEEDS:
        variants = [
            ("dfine_s", ROOT / f"experiments/kamp_v2_baselines/configs/dfine_s_seed{seed}.yml", ROOT / f"experiments/kamp_v2_baselines/runs/dfine_s_seed{seed}/best_stg1.pth", []),
            ("dfine_s_p2", ROOT / f"experiments/kamp_v2_baselines/configs/dfine_s_p2_seed{seed}.yml", ROOT / f"experiments/kamp_v2_baselines/runs/dfine_s_p2_seed{seed}/best_stg1.pth", ["--p2"]),
            ("dfine_M", ROOT / f"experiments/kamp_v2_mal/configs/dfine_M_seed{seed}.yml", ROOT / f"experiments/kamp_v2_mal/runs/dfine_M_seed{seed}/best_stg1.pth", ["--mal"]),
        ]
        for variant, source_config, checkpoint, flags in variants:
            config = yaml.safe_load(source_config.read_text())
            name = f"{variant}_synth25_seed{seed}"
            output = HERE / "runs" / name
            config["output_dir"] = str(output)
            config["epochs"] = 12
            config["checkpoint_freq"] = 12
            config["optimizer"]["lr"] = 5e-5
            config["lr_scheduler"]["milestones"] = [10]
            config["lr_warmup_scheduler"]["warmup_duration"] = 15
            config["train_dataloader"]["dataset"]["img_folder"] = str(DATA / "images/train")
            config["train_dataloader"]["dataset"]["ann_file"] = str(DATA / "annotations/train.json")
            config["val_dataloader"]["dataset"]["img_folder"] = str(DATA / "images/val")
            config["val_dataloader"]["dataset"]["ann_file"] = str(DATA / "annotations/val.json")
            config_path = HERE / "configs" / f"{name}.yml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
            jobs.append({
                "name": name,
                "family": "dfine",
                "variant": variant,
                "seed": seed,
                "config": str(config_path),
                "checkpoint": str(checkpoint),
                "flags": flags,
                "depends_on": None,
            })

        for variant in ["yolov8s", "yolov8s_p2"]:
            name = f"{variant}_synth25_seed{seed}"
            jobs.append({
                "name": name,
                "family": "yolo",
                "variant": variant,
                "seed": seed,
                "checkpoint": str(ROOT / f"experiments/kamp_v2_baselines/runs/{variant}_seed{seed}/weights/best.pt"),
                "depends_on": None,
            })

        jobs.append({
            "name": f"dfine_mal_UQ_synth25_seed{seed}",
            "family": "uq",
            "variant": "dfine_mal_UQ",
            "seed": seed,
            "config": str(HERE / "configs" / f"dfine_M_synth25_seed{seed}.yml"),
            "base_uq_checkpoint": str(ROOT / f"experiments/kamp_v2_uq/runs_frozen/dfine_mal_UQ_seed{seed}/last.pth"),
            "depends_on": f"dfine_M_synth25_seed{seed}",
        })

    hashes = {}
    for path in [SYNTH / "images.csv", SYNTH / "objects.csv", DATA / "annotations/train.json", DATA / "annotations/val.json"]:
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    protocol = {
        "purpose": "supplementary synthetic weakness mitigation; no replacement of frozen final model",
        "real_train_images": 350,
        "synthetic_train_images": len(selected),
        "synthetic_fraction": len(selected) / (350 + len(selected)),
        "synthetic_source_split": "val only",
        "synthetic_test_used_for_training": False,
        "real_validation_background_overlap": "synthetic train derives from the same 66 validation backgrounds; real validation is diagnostic only",
        "epochs": 12,
        "learning_rate": 5e-5,
        "jobs": jobs,
        "quotas": QUOTAS,
        "train_stats": train_stats,
        "val_stats": val_stats,
        "hashes": hashes,
    }
    (HERE / "protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False))
    print(json.dumps({"prepared": True, "train": train_stats, "val": val_stats, "jobs": len(jobs)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
