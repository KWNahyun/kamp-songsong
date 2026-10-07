"""학습용 데이터셋 구축
- data/clean/images : 장비 NG 마커 제거(inpaint) 그레이스케일 PNG   ← 주 실험
- data/marked/images: 마커가 남아 있는 원본 컬러 PNG             ← shortcut 비교용
- data/*/labels     : 공식 TXT 라벨을 그대로 복사 (과제 안내: 마커와 불일치해도 TXT 우선, 수정 금지)
- data/splits/{train,val,test}.txt, split.csv : 세션 그룹 + (장비×결함수) 층화 70/15/15
"""
import os, sys, shutil
import numpy as np
import pandas as pd
import cv2

K = os.environ.get('KAMP_ROOT', '/data/knhyun/KAMP')
sys.argv = ['x']
exec(open(f'{K}/eda/extract.py').read().split('# ─────────────────────────── 1.')[0])

DATA = f'{K}/data'
SEED = 42
RATIOS = {'train': 0.70, 'val': 0.15, 'test': 0.15}

# 데이터 출처: 장비 원본 BMP 폴더 (images 400 은 그중 400장의 바이트 동일 복사본)
raw = pd.read_csv(f'{K}/eda/out/raw_inventory.csv')
stem2rawpath = raw.groupby('stem').path.first().to_dict()
img = pd.read_csv(f'{K}/eda/out/images_with_session.csv')

for v in ['clean', 'marked']:
    for d in ['images', 'labels']:
        os.makedirs(f'{DATA}/{v}/{d}', exist_ok=True)

for s in img.stem:
    path = stem2rawpath[s]
    gray, colored, rgb = load_rgb(path)
    clean = remove_markers(gray, colored)
    cv2.imwrite(f'{DATA}/clean/images/{s}.png', clean)
    cv2.imwrite(f'{DATA}/marked/images/{s}.png', rgb[..., ::-1].astype(np.uint8))   # RGB→BGR
    for v in ['clean', 'marked']:
        shutil.copyfile(f'{SET_DIR}/labels/{s}.txt', f'{DATA}/{v}/labels/{s}.txt')
print(f'이미지 {len(img)}장 저장 완료')

# ── 분할 (v2, 2026-09-29): 그룹 = (장비, 날짜). 같은 장비·같은 날 촬영분은 한 split 에만 → 세션 누수 + 같은 날 조건 공유 차단
#    층 = 장비 × 시편종류(결함 1개 / 3개). 층마다 날짜 배정 조합을 무작위 탐색해 70/15/15 에 가장 가까운 것을 선택
#    (날짜 3개 이상인 층은 val·test 에 최소 1개씩). 이전 세션 분할은 data/splits/split_session_v1.csv
img['kind'] = np.where(img.n_obj == 1, 'k1', 'k3')
img['date'] = img.stem.str[4:12]
img['group'] = img.machine + '_' + img.date
grp = img.groupby('group').agg(machine=('machine', 'first'), kind=('kind', lambda k: k.mode()[0]), n=('stem', 'size')).reset_index()
rng = np.random.default_rng(SEED)
SPL = list(RATIOS)
assign = {}
for (m, k), g in grp.groupby(['machine', 'kind']):
    groups, sizes = g.group.tolist(), g.n.values
    target = np.array([RATIOS[s] for s in SPL]) * sizes.sum()
    best, best_err = None, np.inf
    for _ in range(20000):
        a = rng.integers(0, 3, len(groups))
        if len(groups) >= 3 and not (np.any(a == 1) and np.any(a == 2)):
            continue
        cnt = np.array([sizes[a == i].sum() for i in range(3)])
        err = ((cnt - target) ** 2).sum()
        if err < best_err:
            best, best_err = a, err
    for gname, ai in zip(groups, best):
        assign[gname] = SPL[ai]
img['split'] = img.group.map(assign)

os.makedirs(f'{DATA}/splits', exist_ok=True)
img[['stem', 'machine', 'kind', 'n_obj', 'date', 'group', 'session', 'split', 'labeled_in_400']].to_csv(f'{DATA}/splits/split.csv', index=False)
for v in ['clean', 'marked']:
    for sp in RATIOS:
        stems = img[img.split == sp].stem
        open(f'{DATA}/splits/{v}_{sp}.txt', 'w').write('\n'.join(f'{DATA}/{v}/images/{s}.png' for s in stems) + '\n')
    open(f'{DATA}/{v}.data', 'w').write(
        f'classes=1\ntrain={DATA}/splits/{v}_train.txt\nvalid={DATA}/splits/{v}_val.txt\nnames={DATA}/classes.names\n')
open(f'{DATA}/classes.names', 'w').write('defect\n')

print(pd.crosstab([img.machine, img.kind], img.split, margins=True))
print('(장비,날짜) 그룹 수:', img.groupby('split').group.nunique().to_dict(), '| 세션 수:', img.groupby('split').session.nunique().to_dict())
print('결함 수:', img.groupby('split').n_obj.sum().to_dict())
assert img.groupby('group').split.nunique().max() == 1 and img.groupby('session').split.nunique().max() == 1
print(img.groupby(['split', 'machine']).date.apply(lambda d: sorted(d.unique())).to_string())
