"""Validate the bundled, fixed KAMP split without changing source data."""
from pathlib import Path
import collections
import csv
import hashlib
import json
import math
import struct


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def audit(dataset, reference_manifest):
    dataset = Path(dataset).resolve()
    manifest = dataset / 'split_manifest.csv'
    require(sha256(manifest) == sha256(reference_manifest), 'Split manifest differs from the recorded experiment')
    rows = list(csv.DictReader(manifest.open(encoding='utf-8-sig')))
    require(len({r['image_id'] for r in rows}) == len(rows), 'Duplicate image IDs')
    groups = collections.defaultdict(set)
    seen = {}
    result = {'passed': True, 'manifest_sha256': sha256(manifest), 'splits': {},
              'files': [], 'scope': 'fixed split, groups, PNG dimensions, YOLO/COCO agreement, exact file duplicates'}
    for row in rows:
        require(row['split'] in {'train', 'val', 'test'}, 'Unknown split')
        groups[row['group']].add(row['split'])
    require(all(len(splits) == 1 for splits in groups.values()), 'Equipment/date group leakage')
    for split in ('train', 'val', 'test'):
        subset = [r for r in rows if r['split'] == split]
        require(bool(subset), f'Empty split: {split}')
        coco_path = dataset / 'coco' / f'instances_{split}.json'
        coco = json.loads(coco_path.read_text())
        require(len(coco['categories']) == 1, 'Expected one defect class')
        category = coco['categories'][0]['id']
        images = {Path(im['file_name']).name: im for im in coco['images']}
        require(len(images) == len(coco['images']), 'Duplicate COCO filenames')
        require(len({im['id'] for im in coco['images']}) == len(images), 'Duplicate COCO image IDs')
        expected = {r['image_id'] + '.png' for r in subset}
        require(set(images) == expected, f'{split}: COCO image set differs')
        require({p.name for p in (dataset / 'images' / split).glob('*.png')} == expected,
                f'{split}: missing or extra PNGs')
        anns = collections.defaultdict(list)
        require(len({a['id'] for a in coco['annotations']}) == len(coco['annotations']), 'Duplicate annotation IDs')
        ids = {im['id'] for im in images.values()}
        for ann in coco['annotations']:
            require(ann['image_id'] in ids and ann['category_id'] == category, 'Unknown image/category in COCO')
            anns[ann['image_id']].append(ann['bbox'])
        boxes = 0
        for row in subset:
            path = dataset / row['image_path']
            label = dataset / row['label_path']
            require(path.resolve().is_relative_to(dataset) and label.resolve().is_relative_to(dataset), 'Unsafe data path')
            header = path.read_bytes()[:24]
            require(header[:8] == b'\x89PNG\r\n\x1a\n', f'Invalid PNG: {path}')
            w, h = struct.unpack('>II', header[16:24])
            im = images[path.name]
            require((w, h) == (int(row['width']), int(row['height'])) == (im['width'], im['height']),
                    f'Dimensions differ: {path.name}')
            digest = sha256(path)
            require(digest not in seen or seen[digest] == split, f'Exact duplicate across splits: {path.name}')
            seen[digest] = split
            targets = list(anns[im['id']])
            lines = [line for line in label.read_text().splitlines() if line.strip()]
            require(len(lines) == int(row['n_defects']) == len(targets), f'Box count differs: {path.name}')
            for line in lines:
                cls, cx, cy, bw, bh = map(float, line.split())
                require(cls == 0 and all(math.isfinite(v) for v in (cx, cy, bw, bh)), 'Invalid YOLO values')
                box = [(cx-bw/2)*w, (cy-bh/2)*h, bw*w, bh*h]
                x, y, width, height = box
                require(width > 0 and height > 0 and x >= -0.001 and y >= -0.001
                        and x+width <= w+0.001 and y+height <= h+0.001, 'Box outside image')
                match = next((i for i, target in enumerate(targets)
                              if all(abs(a-b) <= 0.002 for a, b in zip(box, target))), None)
                require(match is not None, f'YOLO/COCO boxes differ: {path.name}')
                targets.pop(match)
            boxes += len(lines)
            result['files'].append({'path': row['image_path'], 'sha256': digest, 'split': split})
        result['splits'][split] = {'images': len(subset), 'boxes': boxes}
    result['groups'] = len(groups)
    return result
