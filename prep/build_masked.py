"""마커 처리 비교 실험용: 마스킹 버전 데이터셋
마커 픽셀(유채색)을 고정 회색값 MASK_VALUE 로 채움 (주변 보간 없음) → 인페인팅(data/clean)과 비교
분할·라벨은 data/clean 과 동일 (공식 TXT 그대로)
출력: data/masked/{images,labels}, data/splits/masked_{train,val,test}.txt, data/masked.data
"""
import os, sys, shutil
import numpy as np
import pandas as pd
import cv2

K = '/data/knhyun/KAMP'
sys.argv = ['x']
exec(open(f'{K}/eda/extract.py').read().split('# ─────────────────────────── 1.')[0])

MASK_VALUE = 128
DATA = f'{K}/data'
raw = pd.read_csv(f'{K}/eda/out/raw_inventory.csv')
stem2rawpath = raw.groupby('stem').path.first().to_dict()
split = pd.read_csv(f'{DATA}/splits/split.csv')

os.makedirs(f'{DATA}/masked/images', exist_ok=True); os.makedirs(f'{DATA}/masked/labels', exist_ok=True)
for s in split.stem:
    gray, colored, _ = load_rgb(stem2rawpath[s])
    out = gray.clip(0, 255).astype(np.uint8)
    out[colored] = MASK_VALUE
    cv2.imwrite(f'{DATA}/masked/images/{s}.png', out)
    shutil.copyfile(f'{DATA}/clean/labels/{s}.txt', f'{DATA}/masked/labels/{s}.txt')
for sp in ['train', 'val', 'test']:
    stems = split[split.split == sp].stem
    open(f'{DATA}/splits/masked_{sp}.txt', 'w').write('\n'.join(f'{DATA}/masked/images/{s}.png' for s in stems) + '\n')
open(f'{DATA}/masked.data', 'w').write(
    f'classes=1\ntrain={DATA}/splits/masked_train.txt\nvalid={DATA}/splits/masked_val.txt\nnames={DATA}/classes.names\n')
print('masked:', len(split), '장')
