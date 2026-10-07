"""모델 담당자 전달용 독립 패키지 생성 → handoff/kamp_xray_v2/ (+ .tar.gz)
포함: 마커 인페인팅 이미지·공식 TXT 라벨(분할별 폴더), split_manifest.csv, data.yaml(YOLO), coco/*.json,
      tools/verify_split.py, tools/evaluate.py, README.md
"""
import os, sys, json, shutil, tarfile
import numpy as np
import pandas as pd
from PIL import Image

K = os.environ.get('KAMP_ROOT', '/data/knhyun/KAMP')
OUT = f'{K}/handoff/kamp_xray_v2'
split = pd.read_csv(f'{K}/data/splits/split.csv')
shutil.rmtree(OUT, ignore_errors=True)
for sp in ['train', 'val', 'test']:
    os.makedirs(f'{OUT}/images/{sp}'); os.makedirs(f'{OUT}/labels/{sp}')
os.makedirs(f'{OUT}/coco'); os.makedirs(f'{OUT}/tools')

rows, coco = [], {sp: dict(images=[], annotations=[], categories=[dict(id=1, name='defect')]) for sp in ['train', 'val', 'test']}
ann_id = 1
for i, r in enumerate(split.itertuples(), 1):
    src_img, src_lab = f'{K}/data/clean/images/{r.stem}.png', f'{K}/data/clean/labels/{r.stem}.txt'
    shutil.copyfile(src_img, f'{OUT}/images/{r.split}/{r.stem}.png')
    shutil.copyfile(src_lab, f'{OUT}/labels/{r.split}/{r.stem}.txt')
    W, H = Image.open(src_img).size
    lines = [l.split() for l in open(src_lab) if l.strip()]
    rows.append(dict(image_id=r.stem, split=r.split, machine={'1호기': 'M1', '2호기': 'M2', '3호기': 'M3'}[r.machine],
                     date=r.date, group=f"{ {'1호기': 'M1', '2호기': 'M2', '3호기': 'M3'}[r.machine] }_{r.date}", session=r.session,
                     n_defects=len(lines), width=W, height=H,
                     image_path=f'images/{r.split}/{r.stem}.png', label_path=f'labels/{r.split}/{r.stem}.txt'))
    coco[r.split]['images'].append(dict(id=i, file_name=f'images/{r.split}/{r.stem}.png', width=W, height=H))
    for c, cx, cy, w, h in lines:
        cx, cy, w, h = float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
        coco[r.split]['annotations'].append(dict(id=ann_id, image_id=i, category_id=1, bbox=[cx - w / 2, cy - h / 2, w, h],
                                                 area=w * h, iscrowd=0))
        ann_id += 1
man = pd.DataFrame(rows)
man.to_csv(f'{OUT}/split_manifest.csv', index=False)
for sp, d in coco.items():
    json.dump(d, open(f'{OUT}/coco/instances_{sp}.json', 'w'))
open(f'{OUT}/data.yaml', 'w').write(
    '# ultralytics YOLO 설정. path 를 이 폴더의 절대 경로로 바꾸세요.\n'
    'path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: defect\n')
for f in ['verify_split.py', 'evaluate.py', 'README.md']:
    src = f'{K}/prep/handoff_src/{f}'
    shutil.copyfile(src, f'{OUT}/{"" if f == "README.md" else "tools/"}{f}')

stat = man.groupby('split').agg(images=('image_id', 'size'), defects=('n_defects', 'sum'), groups=('group', 'nunique'))
print(stat)
assert man.groupby('group').split.nunique().max() == 1
with tarfile.open(f'{K}/handoff/kamp_xray_v2.tar.gz', 'w:gz') as t:
    t.add(OUT, arcname='kamp_xray_v2')
print('패키지:', OUT, '/ 압축:', f'{K}/handoff/kamp_xray_v2.tar.gz')
