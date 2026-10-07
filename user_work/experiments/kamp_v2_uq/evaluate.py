"""Evaluate one frozen-base selector using the common validation protocol."""
from __future__ import annotations

import sys
import json
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
    control=torch.load(checkpoint['base_checkpoint'],map_location='cpu',weights_only=False)
    control_state=control['ema']['module'] if 'ema' in control else control['model']
    assert all(torch.equal(v.cpu(),checkpoint['model'][k].cpu()) for k,v in control_state.items())
    base_kind='base' if '_base_' in name else 'mal'
    seed=name.split('seed')[-1]
    ref_parent=ROOT/'experiments'/('kamp_v2_baselines' if base_kind=='base' else 'kamp_v2_mal')
    ref_name=('dfine_s' if base_kind=='base' else 'dfine_M')+'_seed'+seed
    outdir=HERE/'runs_frozen'/name/'query_audit';outdir.mkdir(exist_ok=True)
    checks=[]

    def predict(unused_name, images):
        for image in images:
            array = np.array(Image.open(common.DATA / "images/val" / image["file_name"]))
            tensor = torch.from_numpy(array.copy()).permute(2, 0, 1)[None].float().cuda() / 255
            with torch.no_grad():
                outputs=model(tensor)
                result = config.postprocessor(outputs, torch.tensor([[640, 640]], device="cuda"))[0]
            boxes=outputs['pred_boxes'][0].cpu().numpy()
            base_scores=outputs['pred_logits_base'][0,:,0].sigmoid().cpu().numpy()
            uq_scores=outputs['pred_logits'][0,:,0].sigmoid().cpu().numpy()
            quality=outputs['pred_quality_logits'][0,:,0].sigmoid().cpu().numpy()
            ref=np.load(ref_parent/'trace'/ref_name/f"image_{image['id']}.npz")
            np.testing.assert_allclose(boxes,ref['decoder_3_boxes'],atol=1e-6,rtol=1e-5)
            np.testing.assert_allclose(base_scores,ref['decoder_3_scores'],atol=1e-6,rtol=1e-5)
            checks.append({'image_id':image['id'],'max_box_diff':float(abs(boxes-ref['decoder_3_boxes']).max()),'max_base_score_diff':float(abs(base_scores-ref['decoder_3_scores']).max())})
            np.savez_compressed(outdir/f"image_{image['id']}.npz",boxes=boxes,base_scores=base_scores,uq_scores=uq_scores,quality=quality)
            keep = result["scores"] >= 0.001
            yield image, result["boxes"][keep].cpu().numpy(), result["scores"][keep].cpu().numpy()

    common.DATA = HERE / "data640"
    common.ORIG = HERE / "original"
    common.EXP = HERE / "runs_frozen"
    common.predict = predict
    common.main(name)
    (outdir/"verification.json").write_text(json.dumps({"base_bitwise_identical":True,"images":checks,"test_used":False},indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: evaluate_frozen.py NAME unary|relation")
    main(sys.argv[1], sys.argv[2])

