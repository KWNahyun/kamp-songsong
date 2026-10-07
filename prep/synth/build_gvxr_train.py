"""gVXR 합성 학습 데이터 (§4.7 다음 단계) — train 사진만 사용, val·test 는 손대지 않음
조건 (모두 tpC 의 700장 = 원본 350 + 옮겨 심기 350 을 그대로 포함하고 350장을 더함 → 1,050장):
  tpCC 학습량 대조 : 옮겨 심기 350장을 한 번 더 (tpC 와 같은 규칙, 다른 난수)
  tpG  gVXR       : train 사진마다 1장. 실제 결함을 지운 배경에 gVXR 이물 1~4개 (제품 안 무작위 자리)
gVXR 이물 (학습에 넣는 범위): 형상 구 / 정육면체 / 불규칙 조각 / 판, 재질 SUS304 / Fe / Al / 유리 / 돌,
  크기 = 같은 부피 구의 지름이 기준 시편(장비별 D_ref) × U_log[0.5, 6], 무작위 회전·부분 픽셀 위치
  대비(배경 × (1 - e^-OD최대)) < 10 회색조면 다시 뽑음 (안 보이는 이물에 라벨을 붙이지 않기 위해)
일반화 확인용으로 학습에서 뺀 것: 선(wire) 형상, 뼈·플라스틱 재질, 기준 × 8 크기 → eval/synth/stress_gvxr.py 로 평가
정답 박스: 이물 영역(OD > 최대의 10%) 외접 사각형 + 2px, 단 장비별 실제 GT 박스 크기 중앙값보다 작지 않게
출력: data/nb2/tp/{tpCC,tpG}/ (kamp_tp.yaml, dfine_train.json, synth_sites.csv) — prep/run_nb2.sh <모델> <gpu> <seed> tpCC|tpG 로 학습
"""
import os, sys, json, zlib, shutil
import numpy as np
import pandas as pd
import cv2
from scipy import ndimage as ndi

K = os.environ.get('KAMP_ROOT', '/data/knhyun/KAMP')
sys.path.insert(0, f'{K}/prep/synth')
import gvxr_core as G
from synth_core import insert
HO = f'{K}/handoff/kamp_xray_v2'; R_T = 7
SRC = f'{K}/data/nb2/tp/tpC'
SHAPES = ['sphere', 'cube', 'irregular', 'plate']; MATS = ['SUS304', 'Fe', 'Al', 'glass', 'stone']
C = pd.read_csv(f'{K}/eval/synth/gvxr/calibration.csv').set_index('machine')


def product_mask(clean):   # eda/extract.py 와 같음
    blur = cv2.GaussianBlur(clean, (9, 9), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(th)
    if n <= 1:
        return np.zeros_like(th, bool)
    return ndi.binary_fill_holes(lab == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))


def yolo_boxes(stem, W, H):
    a = np.array([list(map(float, l.split())) for l in open(f'{HO}/labels/train/{stem}.txt') if l.strip()])
    return np.c_[a[:, 1] * W, a[:, 2] * H, a[:, 3] * W, a[:, 4] * H] if len(a) else np.zeros((0, 4))


split = pd.read_csv(f'{K}/data/splits/split.csv').set_index('stem')
train = split[split.split == 'train'].index.tolist()
P = pd.read_csv(f'{K}/eval/synth/replica_params.csv')
ODMAP = {tuple([k.split('|')[0], int(k.split('|')[1])]): v for k, v in np.load(f'{K}/eval/synth/od_maps.npy', allow_pickle=True).item().items()}
# 기증 결함과 박스 크기 (build_tp_train.py 와 같은 규칙)
donors = {m: [] for m in ['1호기', '2호기', '3호기']}; gtwh = {m: [] for m in donors}
for r in P[P.stem.isin(train)].itertuples():
    im = cv2.imread(f'{HO}/images/train/{r.stem}.png', 0); b = yolo_boxes(r.stem, im.shape[1], im.shape[0])
    j = np.argmin(np.hypot(b[:, 0] - (r.px + 1), b[:, 1] - (r.py + 1))); gtwh[r.machine].append(b[j, 2:4])
    if (r.stem, r.obj) in ODMAP:
        donors[r.machine].append(((r.stem, r.obj), float(b[j, 2]), float(b[j, 3])))
WMIN = {m: np.median(np.array(v), 0) for m, v in gtwh.items()}
print('장비별 최소 박스 (GT 중앙값)', {m: v.round(1).tolist() for m, v in WMIN.items()})
K_TPC = {'1호기': (.3, 1.), '2호기': (.3, 1.), '3호기': (.25, .6)}


def candidates(stem, img):
    H, W = img.shape
    edt = cv2.distanceTransform(product_mask(img).astype(np.uint8), cv2.DIST_L2, 5)
    p = P[P.stem == stem]; real = np.c_[p.px + 1.0, p.py + 1.0]
    yy, xx = np.nonzero(edt >= 2)
    ok = (yy >= R_T) & (xx >= R_T) & (yy + R_T + 2 <= H) & (xx + R_T + 2 <= W)
    if len(real):
        ok &= np.hypot(xx[:, None] + 1 - real[None, :, 0], yy[:, None] + 1 - real[None, :, 1]).min(1) >= 20
    return xx[ok], yy[ok], edt


sim = None
for cond in ['tpCC', 'tpG']:
    d = f'{K}/data/nb2/tp/{cond}'
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(f'{d}/images/train'); os.makedirs(f'{d}/labels/train')
    for f in os.listdir(f'{SRC}/images/train'):          # tpC 700장 그대로 (하드 링크)
        os.link(f'{SRC}/images/train/{f}', f'{d}/images/train/{f}')
        os.link(f'{SRC}/labels/train/{f[:-4]}.txt', f'{d}/labels/train/{f[:-4]}.txt')
    for sp in ['val', 'test']:
        for kind in ['images', 'labels']:
            os.symlink(f'{HO}/{kind}/{sp}', f'{d}/{kind}/{sp}')
    sites = []
    for stem in train:
        img = cv2.imread(f'{HO}/images/train/{stem}.png', 0); H, W = img.shape; m = split.machine[stem]
        rng = np.random.default_rng(zlib.crc32(f'{cond}|{stem}'.encode()))
        bg = cv2.imread(f'{K}/data/synth_val/erased/images/{stem}.png', 0).astype(np.float32)
        xx, yy, edt = candidates(stem, img)
        n = int(rng.integers(1, 5)); Gm = np.zeros((H, W), np.float32); boxes = []; placed = []
        for i in rng.permutation(len(xx)):
            if len(placed) == n:
                break
            x, y = int(xx[i]) + 1, int(yy[i]) + 1
            if cond == 'tpCC':
                if any(np.hypot(x - a, y - b) < 24 for a, b, _ in placed):
                    continue
                (dk, bw, bh) = donors[m][rng.integers(len(donors[m]))]; k = float(rng.uniform(*K_TPC[m])); od = ODMAP[dk]
                Gm[y - 1 - R_T:y - 1 - R_T + od.shape[0], x - 1 - R_T:x - 1 - R_T + od.shape[1]] += k * od
                boxes.append((float(x), float(y), bw, bh)); placed.append((x, y, 8))
                sites.append(dict(stem=stem, machine=m, kind='transplant', x=x, y=y, k=k)); continue
            # gVXR: 대비 ≥ 10 이 될 때까지 다시 뽑음
            if sim is None:
                sim = G.Sim(os_=8, n_px=16)
            c = C.loc[m]
            for _ in range(20):
                shape = SHAPES[rng.integers(4)]; mat = MATS[rng.integers(5)]; scale = float(np.exp(rng.uniform(np.log(.5), np.log(6))))
                od, off = G.make_object(sim, shape, c.D_px * scale, mat, c.p_mm, c.sigma, rng)
                if bg[y, x] * (1 - np.exp(-od.max())) >= 10:
                    break
            else:
                continue
            nn = od.shape[0]; X0, Y0 = x - nn // 2, y - nn // 2
            fy, fx = np.nonzero(od > 0.1 * od.max()); ext = max(fy.ptp(), fx.ptp()) + 1
            if any(np.hypot(x - a, y - b) < (ext + e2) / 2 + 8 for a, b, e2 in placed):
                continue
            ys, xs = slice(max(Y0, 0), min(Y0 + nn, H)), slice(max(X0, 0), min(X0 + nn, W))
            Gm[ys, xs] += od[ys.start - Y0:ys.stop - Y0, xs.start - X0:xs.stop - X0]
            x1, x2 = max(X0 + fx.min() - 2, 0), min(X0 + fx.max() + 3, W); y1, y2 = max(Y0 + fy.min() - 2, 0), min(Y0 + fy.max() + 3, H)
            bw, bh = max(x2 - x1, WMIN[m][0]), max(y2 - y1, WMIN[m][1])
            boxes.append(((x1 + x2) / 2, (y1 + y2) / 2, float(bw), float(bh))); placed.append((x, y, ext))
            sites.append(dict(stem=stem, machine=m, kind='gvxr', x=x, y=y, shape=shape, material=mat, scale=scale,
                              od_peak=float(od.max()), contrast=float(bg[y, x] * (1 - np.exp(-od.max()))), extent=int(ext)))
        out = insert(bg, Gm, OD=1.0); name = f'{stem}__{cond}'
        cv2.imwrite(f'{d}/images/train/{name}.png', out)
        with open(f'{d}/labels/train/{name}.txt', 'w') as f:
            for cx, cy, w, h in boxes:
                f.write(f'0 {cx / W:.6f} {cy / H:.6f} {min(w, W) / W:.6f} {min(h, H) / H:.6f}\n')
    # D-FINE 용 COCO json (전체 train)
    coco = dict(images=[], annotations=[], categories=[dict(id=0, name='defect')])
    for f in sorted(os.listdir(f'{d}/images/train')):
        im = cv2.imread(f'{d}/images/train/{f}', 0); H, W = im.shape; iid = len(coco['images']) + 1
        coco['images'].append(dict(id=iid, file_name=f'images/train/{f}', width=W, height=H))
        for l in open(f'{d}/labels/train/{f[:-4]}.txt'):
            _, cx, cy, w, h = map(float, l.split()); cx, cy, w, h = cx * W, cy * H, w * W, h * H
            coco['annotations'].append(dict(id=len(coco['annotations']) + 1, image_id=iid, category_id=0, bbox=[cx - w / 2, cy - h / 2, w, h], area=w * h, iscrowd=0))
    json.dump(coco, open(f'{d}/dfine_train.json', 'w'))
    open(f'{d}/kamp_tp.yaml', 'w').write(f'# {cond} (prep/synth/build_gvxr_train.py)\npath: {d}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: defect\n')
    S = pd.DataFrame(sites); S.to_csv(f'{d}/synth_sites.csv', index=False)
    print(cond, '사진', len(coco['images']), '박스', len(coco['annotations']), '합성', len(S))
    if cond == 'tpG':
        print(S.groupby(['shape']).size().to_dict(), S.groupby('material').size().to_dict())
        print(S[['scale', 'contrast', 'extent']].describe().round(2))
