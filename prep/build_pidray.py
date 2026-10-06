"""외부 데이터: PIDray → 도메인 사전학습용 YOLO·COCO 데이터 (사전학습 조건 ③: COCO → PIDray → KAMP)
입력: data/external/pidray/annotations/test_easy.json (공식 Google Drive 배포 annotation.tar.gz)
      data/external/pidray/hf/data/*/*.png (Hugging Face Voxel51/PIDray = 공식 test_easy 중 라벨 있는 9,482장)
출력: data/external/pidray_gray/{images,labels}/{train,val}, coco/instances_{train,val}.json, data.yaml, manifest.csv
- 컬러(이중 에너지 의사 색) → 흑백 단채널 PNG (KAMP 영상과 같은 형식). 변환: OpenCV BGR→GRAY (0.299R + 0.587G + 0.114B)
- 12개 클래스 유지 (사전학습에서는 클래스 구분이 특징 학습에 도움. KAMP 미세조정 때 출력 헤드는 1클래스로 새로 초기화)
- 무작위 90 / 10 분할 (사전학습 모니터링용. KAMP 평가와 무관)
라이선스: "ONLY for academic purposes, NOT for commercial purposes" → handoff 패키지에 넣지 않음
출처: Wang et al., ICCV 2021 / Zhang et al., IJCV 2023 (https://github.com/lutao2021/PIDray)
"""
import os, json, glob, shutil
import numpy as np
import pandas as pd
import cv2

K = '/data/knhyun/KAMP'
SRC = f'{K}/data/external/pidray'
OUT = f'{K}/data/external/pidray_gray'
rng = np.random.default_rng(0)

d = json.load(open(f'{SRC}/annotations/test_easy.json'))
cats = sorted(d['categories'], key=lambda c: c['id'])
cid2idx = {c['id']: i for i, c in enumerate(cats)}
files = {os.path.basename(p): p for p in glob.glob(f'{SRC}/hf/data/*/*.png')}
anns = {}
for a in d['annotations']:
    anns.setdefault(a['image_id'], []).append(a)
imgs = [im for im in d['images'] if im['id'] in anns]
missing = [im['file_name'] for im in imgs if im['file_name'] not in files]
print(f'라벨 있는 이미지 {len(imgs)}장, HF에 있는 이미지 {len(files)}장, 없는 이미지 {len(missing)}장')
imgs = [im for im in imgs if im['file_name'] in files]

if os.path.exists(OUT):
    shutil.rmtree(OUT)
order = rng.permutation(len(imgs)); nval = len(imgs) // 10
split = {imgs[j]['id']: ('val' if k < nval else 'train') for k, j in enumerate(order)}
coco = {sp: dict(images=[], annotations=[], categories=[dict(id=i + 1, name=c['name']) for i, c in enumerate(cats)]) for sp in ('train', 'val')}
rows, bad = [], 0
for im in imgs:
    sp = split[im['id']]
    g = cv2.imread(files[im['file_name']], cv2.IMREAD_COLOR)
    if g is None:
        bad += 1; continue
    g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY); h, w = g.shape
    if (h, w) != (im['height'], im['width']):
        bad += 1; continue
    for dd in ('images', 'labels'):
        os.makedirs(f'{OUT}/{dd}/{sp}', exist_ok=True)
    stem = os.path.splitext(im['file_name'])[0]
    cv2.imwrite(f'{OUT}/images/{sp}/{stem}.png', g)
    with open(f'{OUT}/labels/{sp}/{stem}.txt', 'w') as f:
        for a in anns[im['id']]:
            x, y, bw, bh = a['bbox']
            x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + bw, w), min(y + bh, h)
            if x1 <= x0 or y1 <= y0:
                continue
            f.write(f'{cid2idx[a["category_id"]]} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}\n')
            coco[sp]['annotations'].append(dict(id=a['id'], image_id=im['id'], category_id=cid2idx[a['category_id']] + 1,
                                                 bbox=[x0, y0, x1 - x0, y1 - y0], area=(x1 - x0) * (y1 - y0), iscrowd=0))
            rows.append(dict(file=stem, split=sp, cls=cats[cid2idx[a['category_id']]]['name'], w=w, h=h, bw=x1 - x0, bh=y1 - y0))
    coco[sp]['images'].append(dict(id=im['id'], file_name=f'{stem}.png', width=w, height=h))
os.makedirs(f'{OUT}/coco', exist_ok=True)
for sp, c in coco.items():
    json.dump(c, open(f'{OUT}/coco/instances_{sp}.json', 'w'))
names = '\n'.join(f'  {i}: {c["name"]}' for i, c in enumerate(cats))
open(f'{OUT}/data.yaml', 'w').write(f'# PIDray test_easy 라벨 있는 이미지, 흑백 변환 (외부 데이터, 학술 목적만)\npath: {OUT}\ntrain: images/train\nval: images/val\nnames:\n{names}\n')
M = pd.DataFrame(rows); M.to_csv(f'{OUT}/manifest.csv', index=False)
print(f'변환 실패 {bad}장')
print(M.groupby('split').agg(images=('file', 'nunique'), boxes=('file', 'size')))
print('박스 한 변 (px):'); print(pd.concat([M.bw, M.bh]).describe(percentiles=[.05, .5, .95]).round(0))
print('이미지 크기 (h × w):'); print(M.drop_duplicates('file').groupby(['h', 'w']).size().sort_values(ascending=False).head(3))
