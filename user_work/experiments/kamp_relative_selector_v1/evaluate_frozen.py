"""Evaluate one frozen-base selector using the common validation protocol."""
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

    config_name = "dfine_UQ_seed20260929.yml" if mode == "unary" else "dfine_RQS_seed20260929.yml"
    config = YAMLConfig(str(HERE / "configs" / config_name))
    model = config.model
    checkpoint = torch.load(
        HERE / "runs_frozen" / name / "last.pth", map_location="cpu", weights_only=False
    )
    model.load_state_dict(checkpoint["model"])
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

    common.EXP = HERE / "runs_frozen"
    common.predict = predict
    common.main(name)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: evaluate_frozen.py NAME unary|relation")
    main(sys.argv[1], sys.argv[2])

