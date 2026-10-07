"""옮겨 심기 표적 학습 데이터 (§4.6 표적 보강 학습) — train 사진만 사용, val·test 는 손대지 않음
각 조건 = 원본 train 350장(공식 TXT 그대로) + 추가 350장
  tpB 무작위   : train 사진마다 1장. 실제 결함을 지운 배경(data/synth_val/erased)에, 제품 안 무작위 자리 1~4곳(서로 24px 이상, 가장자리 2px 이상,
                 원래 결함 자리와 20px 이상)에 같은 장비 train 결함의 광학 밀도 맵(od_maps.npy)을 배율 k ~ U[0.3, 1.0] 로 곱셈 삽입
                 → 결함이 '띠 끝'에 없는 사진, 띠 끝이 비어 있는 사진을 보여 위치 지름길을 끊는 것이 목적
  tpC 실패 기반: tpB 와 같고, 3호기만 k ~ U[0.25, 0.6] (3호기 저대비 결함, §3.5·§4.4)
  tpX 학습량 대조: 원본 350장을 한 번 더 (합성 없음). 추가 사진 수만 같게 맞춤
합성 결함의 박스: 기증 결함의 공식 TXT 박스 크기(w, h)를 심은 중심에 그대로 둠 (박스 크기도 실제 라벨 분포를 따름)
출력: data/nb2/tp/<조건>/{images,labels}/train (val·test 는 전달 패키지 링크), kamp_tp.yaml (YOLOv8), dfine_train.json (D-FINE), synth_sites.csv
"""
import os, sys, json, zlib
import numpy as np
import pandas as pd
import cv2
from scipy import ndimage as ndi

K = '/data/knhyun/KAMP'
sys.path.insert(0, f'{K}/prep/synth')
from synth_core import insert
HO = f'{K}/handoff/kamp_xray_v2'
R_T = 7
COND = {'tpB': dict(k={'1호기': (.3, 1.), '2호기': (.3, 1.), '3호기': (.3, 1.)}),
        'tpC': dict(k={'1호기': (.3, 1.), '2호기': (.3, 1.), '3호기': (.25, .6)}),
        'tpX': None}


def product_mask(clean):   # eda/extract.py 와 같음
    blur = cv2.GaussianBlur(clean, (9, 9), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(th)
    if n <= 1:
        return np.zeros_like(th, bool)
    return ndi.binary_fill_holes(lab == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))


def yolo_boxes(stem, sp, W, H):
    p = f'{HO}/labels/{sp}/{stem}.txt'
    a = np.array([list(map(float, l.split())) for l in open(p) if l.strip()]) if os.path.exists(p) else np.zeros((0, 5))
    return np.c_[a[:, 1] * W, a[:, 2] * H, a[:, 3] * W, a[:, 4] * H] if len(a) else np.zeros((0, 4))   # cx, cy, w, h (px)


split = pd.read_csv(f'{K}/data/splits/split.csv').set_index('stem')
train = split[split.split == 'train'].index.tolist()
P = pd.read_csv(f'{K}/eval/synth/replica_params.csv')
ODMAP = {tuple([k.split('|')[0], int(k.split('|')[1])]): v for k, v in np.load(f'{K}/eval/synth/od_maps.npy', allow_pickle=True).item().items()}
# 기증 결함 = train 결함 (같은 장비). 박스 크기 = 그 결함 중심에 가장 가까운 공식 TXT 박스
donors = {m: [] for m in ['1호기', '2호기', '3호기']}
for r in P[P.stem.isin(train)].itertuples():
    if (r.stem, r.obj) not in ODMAP:
        continue
    im = cv2.imread(f'{HO}/images/train/{r.stem}.png', 0); b = yolo_boxes(r.stem, 'train', im.shape[1], im.shape[0])
    j = np.argmin(np.hypot(b[:, 0] - (r.px + 1), b[:, 1] - (r.py + 1)))
    donors[r.machine].append(((r.stem, r.obj), float(b[j, 2]), float(b[j, 3])))
print({m: len(v) for m, v in donors.items()})


def link_eval(d):
    for sp in ['val', 'test']:
        for kind in ['images', 'labels']:
            os.makedirs(f'{d}/{kind}', exist_ok=True)
            if not os.path.exists(f'{d}/{kind}/{sp}'):
                os.symlink(f'{HO}/{kind}/{sp}', f'{d}/{kind}/{sp}')


for cond, cfg in COND.items():
    d = f'{K}/data/nb2/tp/{cond}'
    os.makedirs(f'{d}/images/train', exist_ok=True); os.makedirs(f'{d}/labels/train', exist_ok=True); link_eval(d)
    coco = dict(images=[], annotations=[], categories=[dict(id=0, name='defect')]); sites = []

    def add(name, img, boxes):
        cv2.imwrite(f'{d}/images/train/{name}.png', img); H, W = img.shape
        with open(f'{d}/labels/train/{name}.txt', 'w') as f:
            for cx, cy, w, h in boxes:
                f.write(f'0 {cx / W:.6f} {cy / H:.6f} {w / W:.6f} {h / H:.6f}\n')
        iid = len(coco['images']) + 1; coco['images'].append(dict(id=iid, file_name=f'images/train/{name}.png', width=W, height=H))
        for cx, cy, w, h in boxes:
            coco['annotations'].append(dict(id=len(coco['annotations']) + 1, image_id=iid, category_id=0,
                                            bbox=[cx - w / 2, cy - h / 2, w, h], area=w * h, iscrowd=0))

    for stem in train:
        img = cv2.imread(f'{HO}/images/train/{stem}.png', 0); H, W = img.shape
        add(stem, img, yolo_boxes(stem, 'train', W, H))                     # 원본 (공식 TXT 그대로)
        if cfg is None:
            add(f'{stem}__dup', img, yolo_boxes(stem, 'train', W, H)); continue
        rng = np.random.default_rng(zlib.crc32(f'{cond}|{stem}'.encode()))
        m = split.machine[stem]
        bg = cv2.imread(f'{K}/data/synth_val/erased/images/{stem}.png', 0).astype(np.float32)
        edt = cv2.distanceTransform(product_mask(img).astype(np.uint8), cv2.DIST_L2, 5)
        p = P[P.stem == stem]; real = np.c_[p.px + 1.0, p.py + 1.0]
        yy, xx = np.nonzero(edt >= 2)
        ok = (yy >= R_T) & (xx >= R_T) & (yy + R_T + 2 <= H) & (xx + R_T + 2 <= W)
        if len(real):
            ok &= np.hypot(xx[:, None] + 1 - real[None, :, 0], yy[:, None] + 1 - real[None, :, 1]).min(1) >= 20
        yy, xx = yy[ok], xx[ok]
        n = int(rng.integers(1, 5)); ch = []
        for i in rng.permutation(len(xx)):
            if all(np.hypot(xx[i] - a, yy[i] - b) >= 24 for a, b in ch):
                ch.append((int(xx[i]), int(yy[i])))
            if len(ch) == n:
                break
        G = np.zeros((H, W), np.float32); boxes = []
        lo, hi = cfg['k'][m]
        for x, y in ch:
            (dk, bw, bh) = donors[m][rng.integers(len(donors[m]))]; k = float(rng.uniform(lo, hi)); od = ODMAP[dk]
            G[y - R_T:y - R_T + od.shape[0], x - R_T:x - R_T + od.shape[1]] += k * od
            boxes.append((x + 1.0, y + 1.0, bw, bh))
            sites.append(dict(stem=stem, machine=m, x=x, y=y, k=k, donor=f'{dk[0]}|{dk[1]}', edge_dist=float(edt[y + 1, x + 1])))
        add(f'{stem}__{cond}', insert(bg, G, OD=1.0), boxes)
    json.dump(coco, open(f'{d}/dfine_train.json', 'w'))
    open(f'{d}/kamp_tp.yaml', 'w').write(f'# 옮겨 심기 표적 학습 {cond} (prep/synth/build_tp_train.py)\npath: {d}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: defect\n')
    pd.DataFrame(sites).to_csv(f'{d}/synth_sites.csv', index=False)
    print(cond, '사진', len(coco['images']), '박스', len(coco['annotations']), '합성 결함', len(sites))
