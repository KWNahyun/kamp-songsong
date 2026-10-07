from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path("/home/viplab/contest")
HERE = Path(__file__).parent
sys.path[:0] = [str(ROOT / "models/D-FINE"), str(ROOT / "experiments/kamp_v2_uq")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--detector-checkpoint", type=Path, required=True)
    parser.add_argument("--uq-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    from selector_patch import _max_iou_target, install_model
    install_model("unary")
    from src.core import YAMLConfig

    config = YAMLConfig(str(args.config))
    model = config.model.cuda()

    detector = torch.load(args.detector_checkpoint, map_location="cpu", weights_only=False)
    detector_state = detector["ema"]["module"] if "ema" in detector else detector["model"]
    prior_uq = torch.load(args.uq_checkpoint, map_location="cpu", weights_only=False)["model"]
    current = model.state_dict()
    merged = {}
    for key, value in current.items():
        source = prior_uq if "kamp_selector" in key else detector_state
        if key in source and source[key].shape == value.shape:
            merged[key] = source[key]
    missing, unexpected = model.load_state_dict(merged, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")

    for name, parameter in model.named_parameters():
        parameter.requires_grad_("kamp_selector" in name)
    model.eval()
    model.decoder.kamp_selector.train()
    parameters = list(model.decoder.kamp_selector.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=5e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.1)
    train_loader = config.train_dataloader

    args.output.mkdir(parents=True, exist_ok=False)
    log_path = args.output / "log.jsonl"
    started = time.time()
    for epoch in range(12):
        losses = []
        for samples, targets in train_loader:
            samples = samples.cuda(non_blocking=True)
            targets = [{key: value.cuda(non_blocking=True) if torch.is_tensor(value) else value for key, value in target.items()} for target in targets]
            outputs = model(samples)
            target_quality = _max_iou_target(outputs["pred_boxes"].detach(), targets)
            predicted = outputs["pred_quality_logits"].squeeze(-1)
            modulation = (target_quality - predicted.sigmoid()).abs().pow(2)
            loss = (F.binary_cross_entropy_with_logits(predicted, target_quality, reduction="none") * modulation).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if not all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in parameters):
                raise FloatingPointError("Missing or nonfinite UQ gradient")
            optimizer.step()
            losses.append(float(loss.detach()))
        row = {"epoch": epoch + 1, "loss": float(np.mean(losses)), "lr": optimizer.param_groups[0]["lr"], "elapsed_seconds": time.time() - started}
        with log_path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        scheduler.step()

    torch.save({
        "model": model.state_dict(),
        "mode": "unary",
        "seed": args.seed,
        "base_checkpoint": str(args.detector_checkpoint),
        "prior_uq_checkpoint": str(args.uq_checkpoint),
        "base_frozen": True,
        "synthetic_mix": True,
    }, args.output / "last.pth")
    (args.output / "metadata.json").write_text(json.dumps({
        "seed": args.seed,
        "epochs": 12,
        "train_images": len(train_loader.dataset),
        "wall_seconds": time.time() - started,
        "detector_checkpoint": str(args.detector_checkpoint),
        "prior_uq_checkpoint": str(args.uq_checkpoint),
    }, indent=2))


if __name__ == "__main__":
    main()
