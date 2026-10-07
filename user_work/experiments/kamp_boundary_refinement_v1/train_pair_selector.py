"""Train only the base-versus-refined pair selector on frozen MAL+BR."""
from __future__ import annotations

import json
import random
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import torch


ROOT = Path("/home/viplab/contest")
HERE = Path(__file__).parent
sys.path[:0] = [str(ROOT / "models/D-FINE"), str(HERE)]


def main(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    from boundary_patch import install_model

    install_model("box")
    from pair_selector_patch import install_pair_selector, pair_quality_loss

    install_pair_selector()
    from src.core import YAMLConfig

    config = YAMLConfig(str(HERE / "config.yml"))
    model = config.model.cuda()
    source_path = HERE / f"runs/dfine_frozen_BR_seed{seed}/last.pth"
    source = torch.load(source_path, map_location="cpu", weights_only=False)["model"]
    current = model.state_dict()
    matched = {key: value for key, value in source.items() if key in current and value.shape == current[key].shape}
    missing, unexpected = model.load_state_dict(matched, strict=False)
    expected_missing = [key for key in missing if "kamp_pair_selector" in key]
    if sorted(missing) != sorted(expected_missing) or unexpected:
        raise RuntimeError(f"Unexpected checkpoint mismatch: missing={missing}, unexpected={unexpected}")

    for name, parameter in model.named_parameters():
        parameter.requires_grad_("kamp_pair_selector" in name)
    model.eval()
    model.decoder.kamp_pair_selector.train()
    parameters = list(model.decoder.kamp_pair_selector.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[24], gamma=0.1)
    loader = config.train_dataloader

    run_name = f"dfine_frozen_BPS_seed{seed}"
    output = HERE / "runs" / run_name
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    for epoch in range(30):
        losses, refined_rates = [], []
        for samples, targets in loader:
            samples = samples.cuda(non_blocking=True)
            targets = [
                {
                    key: value.cuda(non_blocking=True) if torch.is_tensor(value) else value
                    for key, value in target.items()
                }
                for target in targets
            ]
            outputs = model(samples)
            loss = pair_quality_loss(outputs, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if not all(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in parameters
            ):
                raise FloatingPointError("Missing or nonfinite pair-selector gradient")
            optimizer.step()
            losses.append(float(loss.detach()))
            refined_rates.append(
                float(
                    (
                        outputs["pred_pair_quality_refined"]
                        > outputs["pred_pair_quality_base"]
                    ).float().mean()
                )
            )
        row = {
            "epoch": epoch,
            "train_pair_quality_loss": float(np.mean(losses)),
            "mean_refined_selection_rate": float(np.mean(refined_rates)),
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": time.time() - started,
        }
        with (output / "log.jsonl").open("a") as file:
            file.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        scheduler.step()

    torch.save(
        {
            "model": model.state_dict(),
            "seed": seed,
            "source_checkpoint": str(source_path),
            "mal_and_refiner_frozen": True,
            "validation_used": False,
        },
        output / "last.pth",
    )
    (output / "metadata.json").write_text(
        json.dumps(
            {
                "run": run_name,
                "seed": seed,
                "epochs": 30,
                "source_checkpoint": str(source_path),
                "pair_selector_parameters": sum(parameter.numel() for parameter in parameters),
                "validation_used_for_training_or_selection": False,
                "test_used": False,
                "wall_seconds": time.time() - started,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260929)
    main(parser.parse_args().seed)
