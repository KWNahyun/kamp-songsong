from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


ROOT = Path("/home/viplab/contest")
HERE = Path(__file__).parent
TRAIN = ROOT / "experiments/kamp_synth_train_v1"
FROZEN = ROOT / "experiments/kamp_v2_test_frozen"
sys.path[:0] = [str(ROOT / "experiments/kamp_pilot_v1"), str(ROOT / "models/D-FINE"), str(ROOT / "experiments/kamp_v2_uq")]
from analyze_failures import coco_metrics, match, nms


def load_model(job: dict):
    if job["variant"].startswith("yolo"):
        from ultralytics import YOLO
        return YOLO(job["checkpoint"]), None
    if job["variant"] == "dfine_s_p2":
        import dfine_p2
        dfine_p2.install()
    if job["variant"] == "dfine_mal_UQ":
        from selector_patch import install_model
        install_model("unary")
    from src.core import YAMLConfig
    config = YAMLConfig(job["config"])
    model = config.model
    checkpoint = torch.load(job["checkpoint"], map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"])
    model.cuda().eval()
    return model, config


def predict(job: dict, split: str, model, config) -> tuple[dict, list[dict]]:
    if split == "val":
        ground_truth = json.loads((TRAIN / "data640_mix25/annotations/val.json").read_text())
        image_root = TRAIN / "data640_mix25/images/val"
        transforms = None
    else:
        ground_truth = json.loads((FROZEN / "test_original.json").read_text())
        image_root = FROZEN / "images"
        transforms = json.loads((FROZEN / "transforms.json").read_text())

    predictions = []
    for image in ground_truth["images"]:
        image_path = image_root / image["file_name"]
        if config is None:
            result = model.predict(str(image_path), imgsz=640, device=0, conf=0.001, iou=0.7, max_det=300, rect=False, verbose=False)[0]
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
        else:
            array = np.array(Image.open(image_path)).copy()
            tensor = torch.from_numpy(array).permute(2, 0, 1)[None].float().cuda() / 255
            with torch.inference_mode():
                result = config.postprocessor(model(tensor), torch.tensor([[640, 640]], device="cuda"))[0]
            keep = result["scores"] >= 0.001
            boxes = result["boxes"][keep].cpu().numpy()
            scores = result["scores"][keep].cpu().numpy()
        for box, score in zip(boxes, scores):
            if transforms is None:
                x1, y1, x2, y2 = map(float, box)
            else:
                transform = transforms[image["file_name"]]
                x1, x2 = np.clip((box[[0, 2]].astype(float) - transform["px"]) / transform["sx"], 0, image["width"])
                y1, y2 = np.clip((box[[1, 3]].astype(float) - transform["py"]) / transform["sy"], 0, image["height"])
            if x2 > x1 and y2 > y1:
                predictions.append({"image_id": image["id"], "category_id": 0, "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], "score": float(score)})
    return ground_truth, nms(predictions, 0.7)


def metrics(job: dict, split: str, ground_truth: dict, predictions: list[dict]) -> dict:
    ap = coco_metrics(ground_truth, predictions)
    ground = {image["id"]: [annotation for annotation in ground_truth["annotations"] if annotation["image_id"] == image["id"]] for image in ground_truth["images"]}
    all_events = match(predictions, ground)
    threshold = job["val_threshold"]
    events = [event for event in all_events if event["score"] >= threshold]
    matched = {event["gt_id"] for event in events if event["tp"]}
    tp = len(matched)
    fp = sum(not event["tp"] for event in events)

    # Diagnostic recalibration at FP <= 6 on real validation only.
    recalibrated = None
    if split == "val":
        chosen = {"threshold": 1.000001, "TP": 0, "FP": 0}
        scores = sorted({event["score"] for event in all_events}, reverse=True)
        for candidate in scores:
            current = [event for event in all_events if event["score"] >= candidate]
            current_fp = sum(not event["tp"] for event in current)
            current_tp = len({event["gt_id"] for event in current if event["tp"]})
            if current_fp <= 6:
                chosen = {"threshold": candidate, "TP": current_tp, "FP": current_fp}
            else:
                break
        recalibrated = chosen
    return {
        "name": job["name"], "variant": job["variant"], "seed": job["seed"], "split": split,
        "AP": ap[0] * 100, "AP50": ap[1] * 100, "AP75": ap[2] * 100,
        "locked_threshold": threshold, "TP": tp, "FP": fp, "FN": len(ground_truth["annotations"]) - tp,
        "recall": tp / len(ground_truth["annotations"]), "FP_per_image": fp / len(ground_truth["images"]),
        "recalibrated_FP6": recalibrated,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    args = parser.parse_args()
    protocol = json.loads((HERE / "protocol.json").read_text())
    job = next(item for item in protocol["jobs"] if item["name"] == args.name)
    if hashlib.sha256(Path(job["checkpoint"]).read_bytes()).hexdigest() != job["sha256"]:
        raise RuntimeError("Checkpoint hash mismatch")
    output = HERE / "real" / args.name
    output.mkdir(parents=True, exist_ok=True)
    model, config = load_model(job)
    result = {}
    for split in ["val", "test"]:
        ground_truth, predictions = predict(job, split, model, config)
        result[split] = metrics(job, split, ground_truth, predictions)
        (output / f"{split}_predictions.json").write_text(json.dumps(predictions))
    (output / "metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
