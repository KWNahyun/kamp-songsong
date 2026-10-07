"""마커 제거 방법 비교용 데이터 (GT 500장, 분할은 data/splits/split.csv 그대로, 라벨은 공식 TXT 그대로)
방법:
  rm_ns      : cv2.inpaint Navier-Stokes, 반경 3
  rm_dil     : 마스크를 2px 넓힌 뒤(3×3 팽창 2회) cv2.inpaint Telea, 반경 3
  rm_bgfill  : 마커 픽셀을 주변 정상 픽셀의 가중 평균으로 채움 (정규화 합성곱, Gaussian σ=3)
  rm_biharm  : skimage.restoration.inpaint_biharmonic
  (기존) data/clean = Telea 반경 3, data/masked = 회색 128, data/marked = 원본
출력: data/<방법>/{images,labels}, data/splits/<방법>_{train,val,test}.txt, data/<방법>.data
      eval/marker_v2/variant_check.csv (결함 코어와 제거 영역 사이 최소 거리)
"""
import os, sys, shutil
import numpy as np
import pandas as pd
import cv2

K = '/data/knhyun/KAMP'
sys.path.insert(0, f'{K}/prep')
from rm_methods import METHODS, KER   # 방법 정의는 rm_methods.py
sys.argv = ['x']
exec(open(f'{K}/eda/extract.py').read().split('# ─────────────────────────── 1.')[0])   # load_rgb
DATA = f'{K}/data'
split = pd.read_csv(f'{DATA}/splits/split.csv')
raw = pd.read_csv(f'{K}/eda/out/raw_inventory.csv').groupby('stem').path.first()
obj = pd.read_csv(f'{K}/eda/out/objects.csv')
for v in METHODS:
    for d in ['images', 'labels']:
        os.makedirs(f'{DATA}/{v}/{d}', exist_ok=True)

check = []
for s in split.stem:
    gray, col, _ = load_rgb(raw[s])
    g8 = gray.clip(0, 255).astype(np.uint8); m = col.astype(np.uint8)
    dil = cv2.dilate(m, KER, iterations=2)
    for v, fn in METHODS.items():
        p = f'{DATA}/{v}/images/{s}.png'
        if not os.path.exists(p):
            cv2.imwrite(p, fn(g8, m))
        shutil.copyfile(f'{DATA}/clean/labels/{s}.txt', f'{DATA}/{v}/labels/{s}.txt')
    # 결함 코어(최암점)와 제거 영역 사이 거리 (원래 마스크 / 2px 확장 마스크)
    o = obj[obj.stem == s]
    dt0 = cv2.distanceTransform((1 - m).astype(np.uint8), cv2.DIST_L2, 5)
    dt2 = cv2.distanceTransform((1 - dil).astype(np.uint8), cv2.DIST_L2, 5)
    H, W = g8.shape
    for r in o.itertuples():
        x = int(np.clip(round(r.cx * r.W + r.dark_dx), 0, W - 1)); y = int(np.clip(round(r.cy * r.H + r.dark_dy), 0, H - 1))
        check.append(dict(stem=s, dist_orig=float(dt0[y, x]), dist_dil2=float(dt2[y, x])))

for v in METHODS:
    for sp in ['train', 'val', 'test']:
        stems = split[split.split == sp].stem
        open(f'{DATA}/splits/{v}_{sp}.txt', 'w').write('\n'.join(f'{DATA}/{v}/images/{x}.png' for x in stems) + '\n')
    open(f'{DATA}/{v}.data', 'w').write(
        f'classes=1\ntrain={DATA}/splits/{v}_train.txt\nvalid={DATA}/splits/{v}_val.txt\nnames={DATA}/classes.names\n')
c = pd.DataFrame(check); os.makedirs(f'{K}/eval/marker_v2', exist_ok=True); c.to_csv(f'{K}/eval/marker_v2/variant_check.csv', index=False)
print('결함 코어 ↔ 제거 영역 최소 거리 (px): 원래 마스크', round(c.dist_orig.min(), 1), '/ 2px 확장', round(c.dist_dil2.min(), 1),
      '| 확장 마스크와 2px 이내인 결함:', int((c.dist_dil2 <= 2).sum()), '/', len(c))
