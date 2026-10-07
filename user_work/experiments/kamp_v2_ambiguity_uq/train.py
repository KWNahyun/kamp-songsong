"""Train only the selector on one frozen MAL detector checkpoint."""
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
sys.path[:0] = [str(ROOT / "models/D-FINE"), str(HERE), str(ROOT / "experiments/kamp_v2_uq")]


def main(mode: str, seed: int, base: str, smoke: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    from selector_patch import _max_iou_target, install_model

    install_model(mode)
    from src.core import YAMLConfig

    config_name = "dfine_UQ_seed20260929.yml" if mode == "unary" else "dfine_RQS_seed20260929.yml"
    config = YAMLConfig(str(ROOT / "experiments/kamp_v2_uq/configs" / config_name))
    model = config.model.cuda()
    parent = 'kamp_v2_baselines' if base == 'base' else 'kamp_v2_mal'
    base_name = 'dfine_s' if base == 'base' else 'dfine_M'
    control_path = ROOT / f'experiments/kamp_v2_ambiguity/runs/{base}_seed{seed}/best_stg1.pth'
    control = torch.load(control_path, map_location="cpu", weights_only=False)
    control_state = control["ema"]["module"] if "ema" in control else control["model"]
    current = model.state_dict()
    matched = {key: value for key, value in control_state.items() if key in current and value.shape == current[key].shape}
    missing, unexpected = model.load_state_dict(matched, strict=False)
    expected_missing = [key for key in missing if "kamp_selector" in key]
    if sorted(missing) != sorted(expected_missing) or unexpected:
        raise RuntimeError(f"Unexpected checkpoint mismatch: missing={missing}, unexpected={unexpected}")

    for name, parameter in model.named_parameters():
        parameter.requires_grad_("kamp_selector" in name)
    model.eval()
    model.decoder.kamp_selector.train()
    selector_parameters = list(model.decoder.kamp_selector.parameters())
    optimizer = torch.optim.AdamW(selector_parameters, lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[24], gamma=0.1)
    train_loader = config.train_dataloader

    run_name = f"dfine_frozen_{'UQ' if mode == 'unary' else 'RQS'}_seed{seed}"
    run_name = f'dfine_{base}_UQ_seed{seed}' + ('_smoke' if smoke else '')
    output = HERE / "runs_frozen" / run_name
    output.mkdir(parents=True, exist_ok=False)
    log_path = output / "log.jsonl"
    started = time.time()
    for epoch in range(1 if smoke else 30):
        epoch_losses = []
        for batch_index, (samples, targets) in enumerate(train_loader):
            if smoke and batch_index >= 2: break
            samples = samples.cuda(non_blocking=True)
            targets = [
                {
                    key: value.cuda(non_blocking=True) if torch.is_tensor(value) else value
                    for key, value in target.items()
                }
                for target in targets
            ]
            outputs = model(samples)
            target_quality = _max_iou_target(outputs["pred_boxes"].detach(), targets)
            predicted = outputs["pred_quality_logits"].squeeze(-1)
            modulation = (target_quality - predicted.sigmoid()).abs().pow(2)
            loss = (
                F.binary_cross_entropy_with_logits(predicted, target_quality, reduction="none")
                * modulation
            ).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if not all(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in selector_parameters
            ):
                raise FloatingPointError("Missing or nonfinite selector gradient")
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        row = {
            "epoch": epoch,
            "train_selector_quality": float(np.mean(epoch_losses)),
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": time.time() - started,
        }
        with log_path.open("a") as file:
            file.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        scheduler.step()

    final_state = model.state_dict()
    assert all(torch.equal(value.cpu(), final_state[key].cpu()) for key, value in control_state.items()), 'Frozen detector changed'
    print('FROZEN_DETECTOR_IDENTITY_PASSED', flush=True)
    torch.save(
        {
            "model": model.state_dict(),
            "mode": mode,
            "seed": seed,
            "base_checkpoint": str(control_path),
            "base_frozen": True,
            "validation_used": False,
        },
        output / "last.pth",
    )
    metadata = {
        "run": run_name,
        "mode": mode,
        "seed": seed,
        "epochs": 1 if smoke else 30,
        "base_bitwise_identical": True,
        "base_kind": base,
        "base_checkpoint": str(control_path),
        "base_frozen": True,
        "selector_parameters": sum(parameter.numel() for parameter in selector_parameters),
        "train_images": len(train_loader.dataset),
        "validation_used_for_training_or_selection": False,
        "test_used": False,
        "wall_seconds": time.time() - started,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["unary", "relation"], required=True)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--base", choices=["mal", "edge", "soft10"], required=True)
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    main(arguments.mode, arguments.seed, arguments.base, arguments.smoke)
