"""Preflight for identity, target assignment, gradient isolation, and VRAM."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


ROOT = Path("/home/viplab/contest")
HERE = Path(__file__).parent
sys.path[:0] = [str(ROOT / "models/D-FINE"), str(HERE)]


def main() -> None:
    from boundary_patch import install_model, refinement_loss

    install_model("edge")
    from src.core import YAMLConfig

    config = YAMLConfig(str(HERE / "config.yml"))
    model = config.model.cuda().eval()
    checkpoint = torch.load(
        ROOT / "experiments/kamp_ablation_v3/runs/dfine_M_seed20260929/best_stg1.pth",
        map_location="cpu",
        weights_only=False,
    )
    control = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    current = model.state_dict()
    matched = {key: value for key, value in control.items() if key in current and value.shape == current[key].shape}
    model.load_state_dict(matched, strict=False)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_("kamp_boundary_refiner" in name)
    model.decoder.kamp_boundary_refiner.train()

    samples, targets = next(iter(config.train_dataloader))
    samples = samples.cuda()
    targets = [
        {key: value.cuda() if torch.is_tensor(value) else value for key, value in target.items()}
        for target in targets
    ]
    outputs = model(samples)
    initial_difference = float(
        (outputs["pred_boxes"] - outputs["pred_boxes_base"]).abs().max().detach()
    )
    if initial_difference > 2e-7:
        raise AssertionError(f"Zero-init refiner changed boxes by {initial_difference}")
    loss, diagnostics = refinement_loss(outputs, targets)
    base_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if "kamp_boundary_refiner" not in name and parameter.requires_grad
    ]
    if base_parameters:
        raise AssertionError("Base detector parameters were not frozen")
    loss.backward()
    refiner_gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if "kamp_boundary_refiner" in name and parameter.requires_grad
    ]
    finite = all(gradient is not None and torch.isfinite(gradient).all() for gradient in refiner_gradients)
    nonzero = sum(float(gradient.abs().sum()) for gradient in refiner_gradients if gradient is not None) > 0
    if not finite or not nonzero:
        raise AssertionError("Refiner gradients are missing, zero, or nonfinite")

    report = {
        "passed": True,
        "mode": "edge",
        "split": "train",
        "test_used": False,
        "batch_shape": list(samples.shape),
        "zero_init_max_box_difference": initial_difference,
        "base_parameters_trainable": 0,
        "refiner_parameters": sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if "kamp_boundary_refiner" in name
        ),
        "positive_queries": diagnostics["positive_queries"],
        "mean_selected_base_iou": diagnostics["mean_base_iou"],
        "initial_loss": float(loss.detach()),
        "refiner_gradients_finite": finite,
        "refiner_gradient_nonzero": nonzero,
        "peak_vram_MiB": torch.cuda.max_memory_allocated() / 2**20,
    }
    (HERE / "preflight.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
