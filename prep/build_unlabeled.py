"""GT 없는 데이터 준비: 이름 부여 → 마커 인페인팅 이미지 → 역할 배정
입력: eda/out/unlabeled_images.csv (eda/unlabeled.py), eda/out/raw_inventory.csv (eda/extract.py), data/splits/split.csv (prep/build_dataset.py)
출력: <img_dir>/<name>.png   (기본 data/unlabeled/images, GT 와 같은 인페인팅: TELEA, 반경 3)
      <roles>                 (기본 data/splits/unlabeled_roles.csv)
역할 (GT 분할 기준, 누수 방지):
  - 이상            : 제품 잘림·없음, "T" 표시, 제품 밖 표시 (eda/unlabeled.py 의 품질 판정) → 학습 제외, 헛검출 점검용
  - 검증그룹(제외)   : val·test GT 와 같은 (장비, 날짜) → 어디에도 쓰지 않음
  - PL학습          : train GT 와 같은 (장비, 날짜) + GT 없는 날짜의 70% (장비별, 날짜 단위, seed 42)
  - PL평가          : GT 없는 날짜의 30% → 처음 보는 날짜에서의 검출 일관성 (참고 지표)
파일명이 같고 내용이 다른 3건은 두 번째 버전에 '_v2' 를 붙여 구분
사용: python3 prep/build_unlabeled.py [--img-dir DIR] [--roles FILE]
"""
import os, sys, argparse
import numpy as np
import pandas as pd
import cv2

K = '/data/knhyun/KAMP'
ap = argparse.ArgumentParser()
ap.add_argument('--img-dir', default=f'{K}/data/unlabeled/images')
ap.add_argument('--roles', default=f'{K}/data/splits/unlabeled_roles.csv')
args = ap.parse_args()
sys.argv = ['x']
exec(open(f'{K}/eda/extract.py').read().split('# ─────────────────────────── 1.')[0])   # load_rgb, remove_markers

SEED, EVAL_FRAC = 42, 0.30
os.makedirs(args.img_dir, exist_ok=True)

u = pd.read_csv(f'{K}/eda/out/unlabeled_images.csv')
raw = pd.read_csv(f'{K}/eda/out/raw_inventory.csv').drop_duplicates('md5').set_index('md5')
split = pd.read_csv(f'{K}/data/splits/split.csv')

meta = u[u.use == '미라벨'].copy()
meta['name'] = meta.stem + np.where(meta.duplicated('stem', keep='first'), '_v2', '')

# ── 인페인팅 이미지
n_new = 0
for _, r in meta.iterrows():
    p = f'{args.img_dir}/{r["name"]}.png'
    if not os.path.exists(p):
        gray, col, _ = load_rgb(raw.path[r.md5])
        cv2.imwrite(p, remove_markers(gray, col)); n_new += 1
print(f'인페인팅 이미지: {len(meta)}장 (새로 생성 {n_new})')

# ── 역할 배정
meta['group'] = meta.machine + '_' + meta.stem.str[4:12]
meta['gt_split'] = meta.group.map(split.groupby('group').split.first().to_dict())
meta['role'] = ''
meta.loc[meta.quality != '정상', 'role'] = '이상'
meta.loc[(meta.role == '') & meta.gt_split.isin(['val', 'test']), 'role'] = '검증그룹(제외)'
meta.loc[(meta.role == '') & (meta.gt_split == 'train'), 'role'] = 'PL학습'
rng = np.random.default_rng(SEED)
free = meta[(meta.role == '') & meta.gt_split.isna()]
for m, g in free.groupby('machine'):
    dates = np.array(sorted(g.date.unique()))
    ev = set(rng.choice(dates, int(round(len(dates) * EVAL_FRAC)), replace=False))
    meta.loc[g.index, 'role'] = np.where(g.date.isin(ev), 'PL평가', 'PL학습')
assert (meta.role != '').all()
cols = ['name', 'stem', 'md5', 'machine', 'date', 'session', 'w', 'h', 'quality', 'n_marker', 'group', 'gt_split', 'role']
meta[cols].to_csv(args.roles, index=False)
print(pd.crosstab(meta.machine, meta.role, margins=True))
