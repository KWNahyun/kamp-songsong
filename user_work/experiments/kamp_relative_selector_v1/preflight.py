"""Preflight checks for the detached relative-selector experiment."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


ROOT = Path("/home/viplab/contest")
HERE = Path(__file__).parent
sys.path[:0] = [
    str(ROOT / "models/D-FINE"),
    str(ROOT / "experiments/kamp_ablation_v3"),
    str(HERE),
]


def matched_load(model: torch.nn.Module, state: dict) -> None:
    current = model.state_dict()
    matched = {k: v for k, v in state.items() if k in current and v.shape == current[k].shape}
    model.load_state_dict(matched, strict=False)


def move_targets(targets: list[dict], device: str) -> list[dict]:
    return [
        {key: value.to(device) if torch.is_tensor(value) else value for key, value in target.items()}
        for target in targets
    ]


def main() -> None:
    from src.core import YAMLConfig
    from loss_patch import install as install_mal

    config_path = HERE / "configs/dfine_RQS_seed20260929.yml"
    checkpoint = torch.load(
        ROOT / "experiments/kamp_pilot_v1/dfine_s_init.pth",
        map_location="cpu",
        weights_only=False,
    )
    initial_state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]

    # Obtain the unpatched MAL output first; class-level patching occurs below.
    install_mal(0.0, True)
    base_cfg = YAMLConfig(str(config_path))
    base_model = base_cfg.model.cuda().train()
    matched_load(base_model, initial_state)
    samples, targets = next(iter(base_cfg.train_dataloader))
    samples = samples.cuda()
    targets = move_targets(targets, "cuda")
    torch.manual_seed(4401)
    with torch.no_grad():
        base_output = base_model(samples, targets=targets)
        base_logits = base_output["pred_logits"].clone()
        base_boxes = base_output["pred_boxes"].clone()
    del base_output, base_model
    torch.cuda.empty_cache()

    from selector_patch import install as install_selector

    install_selector("relation", quality_weight=1.0, criterion=True)
    selector_cfg = YAMLConfig(str(config_path))
    model = selector_cfg.model.cuda().train()
    criterion = selector_cfg.criterion.cuda()
    matched_load(model, initial_state)
    torch.manual_seed(4401)
    output = model(samples, targets=targets)

    detector_logits_unchanged = torch.equal(base_logits, output["pred_logits"])
    detector_boxes_unchanged = torch.equal(base_boxes, output["pred_boxes"])
    if not detector_logits_unchanged or not detector_boxes_unchanged:
        raise AssertionError("Selector altered the MAL detector forward path")

    losses = criterion(output, targets)
    selector_loss = losses["loss_selector_quality"]
    base_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if "kamp_selector" not in name and parameter.requires_grad
    ]
    detached_gradients = torch.autograd.grad(
        selector_loss,
        base_parameters,
        retain_graph=True,
        allow_unused=True,
    )
    selector_detached = all(gradient is None for gradient in detached_gradients)
    if not selector_detached:
        raise AssertionError("Selector loss leaked gradients into the detector")

    total_loss = sum(losses.values())
    total_loss.backward()
    selector_gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if "kamp_selector" in name and parameter.requires_grad
    ]
    selector_gradient_finite = all(
        gradient is not None and torch.isfinite(gradient).all() for gradient in selector_gradients
    )
    selector_gradient_nonzero = sum(
        float(gradient.abs().sum()) for gradient in selector_gradients if gradient is not None
    ) > 0
    if not selector_gradient_finite or not selector_gradient_nonzero:
        raise AssertionError("Selector did not receive a finite, nonzero gradient")

    report = {
        "passed": True,
        "split": "train",
        "test_used": False,
        "batch_shape": list(samples.shape),
        "detector_logits_bitwise_unchanged": detector_logits_unchanged,
        "detector_boxes_bitwise_unchanged": detector_boxes_unchanged,
        "selector_loss_detached_from_detector": selector_detached,
        "selector_gradient_finite": selector_gradient_finite,
        "selector_gradient_nonzero": selector_gradient_nonzero,
        "selector_parameters": sum(
            parameter.numel() for name, parameter in model.named_parameters() if "kamp_selector" in name
        ),
        "selector_loss_initial": float(selector_loss.detach()),
        "total_loss_initial": float(total_loss.detach()),
        "peak_vram_MiB": torch.cuda.max_memory_allocated() / 2**20,
    }
    (HERE / "preflight.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

