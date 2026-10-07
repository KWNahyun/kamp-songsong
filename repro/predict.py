"""MAL+UQ inference on marker-removed grayscale images. No ground truth is read."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variant', choices=['dfine_mal_UQ'], default='dfine_mal_UQ')
    parser.add_argument('--seed', type=int, default=20260930)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--checkpoint', type=Path, help='Explicit retrained MAL+UQ checkpoint')
    parser.add_argument('--checkpoint-sha256', help='Required hash for an explicit checkpoint')
    parser.add_argument('--preprocessed', action='store_true', required=True,
                        help='Confirm marker-removed grayscale input, not marked originals')
    args = parser.parse_args()
    if not args.images.exists():
        raise FileNotFoundError(args.images)
    paths = sorted([args.images] if args.images.is_file() else
                   [p for p in args.images.iterdir() if p.suffix.lower() in ('.png', '.bmp', '.jpg', '.jpeg')])
    if not paths:
        raise ValueError('No input images')
    if args.output.exists():
        raise FileExistsError('Use a new output directory to preserve earlier outputs')
    manifest = json.loads((ROOT / 'manifests/model.json').read_text())
    frozen = args.checkpoint is None
    if frozen and args.seed != manifest['seed']:
        raise ValueError('The frozen representative checkpoint requires seed 20260930')
    checkpoint = args.checkpoint.resolve() if args.checkpoint else ROOT / manifest['checkpoint']
    expected = args.checkpoint_sha256 if args.checkpoint else manifest['sha256']
    if not expected:
        raise ValueError('--checkpoint-sha256 is required for a retrained checkpoint')
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != expected:
        raise ValueError(f'Checkpoint SHA256 mismatch: expected={expected}, actual={digest}')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Fixed-model inference supports --device cpu')
    sys.path[:0] = [str(ROOT / 'vendor/dfine'), str(ROOT / 'src')]
    from metrics import nms
    from selector_patch import install_model
    install_model('unary')
    from src.core import YAMLConfig
    torch.set_num_threads(4)
    config = YAMLConfig(str(ROOT / manifest['config']))
    model = config.model
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if not frozen and state.get('seed', args.seed) != args.seed:
        raise ValueError('Retrained checkpoint seed differs from requested seed')
    model.load_state_dict(state['ema']['module'] if 'ema' in state else state['model'])
    model.to(device).eval()
    args.output.mkdir(parents=True)
    (args.output / 'overlays').mkdir()
    started = time.time()
    allpred, summaries = [], []
    for image_id, path in enumerate(paths, 1):
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None or image.dtype != np.uint8:
            raise ValueError(f'Expected decodable uint8 image: {path}')
        if image.ndim == 3:
            if image.shape[2] != 3 or not np.array_equal(image[:, :, 0], image[:, :, 1]) or not np.array_equal(image[:, :, 0], image[:, :, 2]):
                raise ValueError(f'Marked/color image is outside the preprocessing contract: {path}')
            gray = image[:, :, 0]
        elif image.ndim == 2:
            gray = image
        else:
            raise ValueError(f'Unexpected channels: {path}')
        h, w = gray.shape
        nw, nh = round(w * 640 / max(w, h)), round(h * 640 / max(w, h))
        px, py = (640 - nw) // 2, (640 - nh) // 2
        canvas = np.full((640, 640, 3), 114, np.uint8)
        canvas[py:py+nh, px:px+nw] = cv2.resize(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), (nw, nh), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(canvas.copy()).permute(2, 0, 1)[None].float().to(device) / 255
        with torch.no_grad():
            output = config.postprocessor(model(tensor), torch.tensor([[640, 640]], device=device))[0]
        valid = output['scores'] >= manifest['score_floor']
        boxes = output['boxes'][valid].cpu().numpy()
        scores = output['scores'][valid].cpu().numpy()
        predictions = []
        for box, score in zip(boxes, scores):
            xx = np.clip((box[[0, 2]].astype(float) - px) / (nw / w), 0, w)
            yy = np.clip((box[[1, 3]].astype(float) - py) / (nh / h), 0, h)
            if not np.isfinite(box).all() or not np.isfinite(score):
                raise FloatingPointError('Nonfinite model output')
            if xx[1] > xx[0] and yy[1] > yy[0]:
                predictions.append(dict(image_id=image_id, file_name=path.name, category_id=0,
                    bbox=[float(xx[0]), float(yy[0]), float(xx[1]-xx[0]), float(yy[1]-yy[0])], score=float(score)))
        predictions = nms(predictions, manifest['NMS'])
        alarms = [p for p in predictions if p['score'] >= manifest['val_threshold']]
        allpred.extend(predictions)
        overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for pred in alarms:
            x, y, bw, bh = pred['bbox']
            cv2.rectangle(overlay, (round(x), round(y)), (round(x+bw), round(y+bh)), (0, 130, 255), 1)
            cv2.putText(overlay, f"{pred['score']:.3f}", (max(0, round(x)), max(10, round(y)-3)), cv2.FONT_HERSHEY_SIMPLEX, .3, (0, 80, 220), 1, cv2.LINE_AA)
        if not cv2.imwrite(str(args.output / 'overlays' / f'{image_id:04d}_{path.stem}.png'), overlay):
            raise OSError(f'Could not save overlay for {path}')
        summary = dict(image_id=image_id, file_name=path.name, width=w, height=h,
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), threshold=manifest['val_threshold'],
            detections=len(alarms), status='detections_present' if alarms else 'no_detection_review_required',
            auto_pass_authorized=False)
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    (args.output / 'predictions.json').write_text(json.dumps(allpred, indent=2))
    (args.output / 'image_summary.json').write_text(json.dumps(summaries, indent=2))
    metadata = dict(run=manifest['name'] if frozen else f'dfine_mal_UQ_retrained_seed{args.seed}',
        frozen_representative=frozen, seed=args.seed, checkpoint_sha256=digest, device=str(device),
        val_threshold=manifest['val_threshold'], threshold_policy='fixed original validation threshold; not reoptimized',
        images=len(paths), seconds=time.time()-started, GT_used=False,
        preprocessing='provided marker-removed uint8 grayscale; original-resolution input; 640 central letterbox',
        NMS=manifest['NMS'], box_scale=1, score_floor=manifest['score_floor'], score_is_calibrated_probability=False)
    (args.output / 'run_metadata.json').write_text(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
