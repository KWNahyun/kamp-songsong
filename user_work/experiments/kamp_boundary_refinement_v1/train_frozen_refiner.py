"""Train one boundary refiner while the selected MAL detector stays frozen."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path("/home/viplab/contest")
HERE = Path(__file__).parent
sys.path[:0] = [str(ROOT / "models/D-FINE"), str(HERE)]


def main(mode: str, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    from boundary_patch import install_model, refinement_loss

    install_model(mode)
    from src.core import YAMLConfig

    config = YAMLConfig(str(HERE / "config.yml"))
    model = config.model.cuda()
    control_path = (
        ROOT / f"experiments/kamp_ablation_v3/runs/dfine_M_seed{seed}/best_stg1.pth"
    )
    control = torch.load(control_path, map_location="cpu", weights_only=False)
    control_state = control["ema"]["module"] if "ema" in control else control["model"]
    current = model.state_dict()
    matched = {
        key: value
        for key, value in control_state.items()
        if key in current and value.shape == current[key].shape
    }
    missing, unexpected = model.load_state_dict(matched, strict=False)
    expected_missing = [key for key in missing if "kamp_boundary_refiner" in key]
    if sorted(missing) != sorted(expected_missing) or unexpected:
        raise RuntimeError(f"Unexpected checkpoint mismatch: missing={missing}, unexpected={unexpected}")

    for name, parameter in model.named_parameters():
        parameter.requires_grad_("kamp_boundary_refiner" in name)
    model.eval()
    model.decoder.kamp_boundary_refiner.train()
    parameters = list(model.decoder.kamp_boundary_refiner.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[24], gamma=0.1)
    train_loader = config.train_dataloader

    label = "BR" if mode == "box" else "EBR"
    run_name = f"dfine_frozen_{label}_seed{seed}"
    output = HERE / "runs" / run_name
    output.mkdir(parents=True, exist_ok=False)
    log_path = output / "log.jsonl"
    started = time.time()
    for epoch in range(30):
        losses, positives, base_ious = [], [], []
        for samples, targets in train_loader:
            samples = samples.cuda(non_blocking=True)
            targets = [
                {
                    key: value.cuda(non_blocking=True) if torch.is_tensor(value) else value
                    for key, value in target.items()
                }
                for target in targets
            ]
            outputs = model(samples)
            loss, diagnostics = refinement_loss(outputs, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if not all(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in parameters
            ):
                raise FloatingPointError("Missing or nonfinite refiner gradient")
            optimizer.step()
            losses.append(float(loss.detach()))
            positives.append(diagnostics["positive_queries"])
            base_ious.append(diagnostics["mean_base_iou"])
        row = {
            "epoch": epoch,
            "train_refinement_loss": float(np.mean(losses)),
            "mean_positive_queries_per_batch": float(np.mean(positives)),
            "mean_selected_base_iou": float(np.mean(base_ious)),
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": time.time() - started,
        }
        with log_path.open("a") as file:
            file.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        scheduler.step()

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
        "epochs": 30,
        "base_checkpoint": str(control_path),
        "base_frozen": True,
        "refiner_parameters": sum(parameter.numel() for parameter in parameters),
        "max_edge_shift_relative_to_base_size": 0.75,
        "topk_candidates_per_gt": 10,
        "minimum_assignment_iou": 0.1,
        "validation_used_for_training_or_selection": False,
        "test_used": False,
        "wall_seconds": time.time() - started,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["box", "edge"], required=True)
    parser.add_argument("--seed", type=int, default=20260929)
    arguments = parser.parse_args()
    main(arguments.mode, arguments.seed)

