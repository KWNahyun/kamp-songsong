"""Validation-only common evaluation for selector checkpoints."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


ROOT = Path("/home/viplab/contest")
HERE = Path(__file__).parent
PILOT = ROOT / "experiments/kamp_pilot_v1"
sys.path[:0] = [str(PILOT), str(ROOT / "models/D-FINE"), str(HERE)]

import evaluate_common as common


def main(name: str, mode: str) -> None:
    from selector_patch import install_model

    install_model(mode)
    from src.core import YAMLConfig

    config = YAMLConfig(str(HERE / "configs" / f"{name}.yml"))
    model = config.model
    checkpoint = torch.load(
        HERE / "runs" / name / "best_stg1.pth", map_location="cpu", weights_only=False
    )
    model.load_state_dict(checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"])
    model.cuda().eval()

    def predict(unused_name, images):
        for image in images:
            array = np.array(Image.open(common.DATA / "images/val" / image["file_name"]))
            tensor = torch.from_numpy(array.copy()).permute(2, 0, 1)[None].float().cuda() / 255
            with torch.no_grad():
                result = config.postprocessor(
                    model(tensor), torch.tensor([[640, 640]], device="cuda")
                )[0]
            keep = result["scores"] >= 0.001
            yield image, result["boxes"][keep].cpu().numpy(), result["scores"][keep].cpu().numpy()

    common.EXP = HERE / "runs"
    common.predict = predict
    common.main(name)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: evaluate_selectors.py NAME unary|relation")
    main(sys.argv[1], sys.argv[2])

