"""스트레스: 크기·형상·재질 축 — gVXR 로 만든 이물을 실제 배경(결함을 지운 val·test 150장)의 제품 안 무작위 자리에 곱셈 삽입 (학습 없음)
이물 모형 (prep/synth/gvxr_core.py): 3D 형상 투영 두께 × 재질 μ(E) × 다색 스펙트럼(70 kVp, Al 2 mm 가정) → 검출기 흐림 → 광학 밀도
보정 (eval/synth/gvxr/calibration.csv): 실제 결함 = SUS304 구 로 보고 장비별 픽셀 크기 p(약 0.1 mm/px)와 기준 지름 D_ref(약 2.2~2.5 px, 약 0.25 mm)
축 (한 번에 하나만 바꿈):
  size     : SUS304 구, 지름 = D_ref × {0.5, 0.75, 1, 1.5, 2, 3, 5}
  shape    : SUS304, 부피 = 기준 구(D_ref) 와 같음, 무작위 회전. 구 / 정육면체 / 불규칙 조각 / 판(두께 = 한 변의 1/5) / 선(길이 = 지름의 10배, 20배)
  material : 구, 지름 = D_ref × {1, 2, 4, 8}, 재질 = SUS304 / Al / 유리 / 돌 / 뼈 / 플라스틱
자리: stress_pos.py 와 같은 규칙(같은 난수)으로 이미지마다 3곳 × 2회. 각 자리의 이물은 자리마다 다른 회전·부분 픽셀 위치
검출: 신뢰도 ≥ 0.25 인 검출 중심이 이물 중심 8px 안이거나 이물 영역(OD > 최대의 5%)을 4px 넓힌 곳 안 → 찾음 (선처럼 긴 이물 고려)
모델: 한 프로세스에서 여러 모델을 함께 (같은 영상): YOLOv3 B_s0~2 (416) + 인자로 준 새 모델 변형 seed 3개씩 (640)
출력: eval/synth/gvxr/stress_site{GVXR_TAG}.csv (조건 × 자리 × 모델: 신뢰도, 찾음, 최대 OD, 대비, 투영 면적), stress_fp.csv
"""
import os, sys, zlib
import numpy as np
import pandas as pd
import cv2
from scipy import ndimage as ndi

K = '/data/knhyun/KAMP'
sys.path.insert(0, f'{K}/prep/synth'); sys.path.insert(0, f'{K}/eval')
VARS = sys.argv[1:] or ['yolov8s', 'yolov8s_tpB', 'yolov8s_tpC', 'dfine_s', 'dfine_s_tpC']; sys.argv = ['x']
import gvxr_core as G
from synth_core import insert
import evaluate as E
import nb2_models as NB
NB.patch(E)
OUT = f'{K}/eval/synth/gvxr'
C = pd.read_csv(f'{OUT}/calibration.csv').set_index('machine')
NPASS, NSITE, R_T = 2, 3, 7
COND = [('size', f'x{k}', dict(shape='sphere', mat='SUS304', scale=k)) for k in (0.5, 0.75, 1, 1.5, 2, 3, 5)] + \
       [('shape', s, dict(shape=s, mat='SUS304', scale=1)) for s in ('sphere', 'cube', 'irregular', 'plate', 'wire10', 'wire20')] + \
       [('material', f'{m}_x{k}', dict(shape='sphere', mat=m, scale=k)) for m in ('SUS304', 'Al', 'glass', 'stone', 'bone', 'plastic') for k in (1, 2, 4, 8)]


def product_mask(clean):   # eda/extract.py 와 같음
    blur = cv2.GaussianBlur(clean, (9, 9), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(th)
    if n <= 1:
        return np.zeros_like(th, bool)
    return ndi.binary_fill_holes(lab == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))


split = pd.read_csv(f'{K}/data/splits/split.csv').set_index('stem')
vt = split[split.split.isin(['val', 'test'])].index.tolist()
P = pd.read_csv(f'{K}/eval/synth/replica_params.csv')


def sites_for(stem):
    """stress_pos.sites_for 와 같은 자리 (같은 난수 순서; 기증 결함 선택에 쓰던 난수도 그대로 소비)"""
    rng = np.random.default_rng(zlib.crc32(stem.encode()))
    clean = cv2.imread(f'{K}/data/clean/images/{stem}.png', 0); H, W = clean.shape
    pm = product_mask(clean); edt = cv2.distanceTransform(pm.astype(np.uint8), cv2.DIST_L2, 5)
    p = P[P.stem == stem]; real = np.c_[p.px + 1.0, p.py + 1.0]
    yy, xx = np.nonzero(edt >= 2)
    ok = (yy >= R_T) & (xx >= R_T) & (yy + R_T + 2 <= H) & (xx + R_T + 2 <= W)
    ok &= np.hypot(xx[:, None] + 1 - real[None, :, 0], yy[:, None] + 1 - real[None, :, 1]).min(1) >= 20
    yy, xx = yy[ok], xx[ok]
    passes = []
    for _ in range(4):
        ch = []
        for i in rng.permutation(len(xx)):
            if all(np.hypot(xx[i] - a, yy[i] - b) >= 24 for a, b in ch):
                ch.append((int(xx[i]) + 1, int(yy[i]) + 1)); rng.integers(1000)
            if len(ch) == NSITE:
                break
        passes.append(ch)
    return edt, passes[:NPASS]


sim = G.Sim(os_=8, n_px=16)


def object_od(cond, machine, rng):
    c = C.loc[machine]; D = c.D_px * cond['scale']; V = np.pi / 6 * c.D_px ** 3     # 기준 부피 (px³)
    rot = tuple(rng.uniform(0, 360, 3)); off = tuple(rng.uniform(0, 1, 2))
    s = cond['shape']
    if s == 'sphere':
        args, ext = ('sphere', D), D
    elif s == 'cube':
        a = V ** (1 / 3); args, ext = ('cube', a), a * 1.8
    elif s == 'irregular':
        args, ext = ('irregular', D), D * 2.2
    elif s == 'plate':
        a = (5 * V) ** (1 / 3); args, ext = ('plate', (a, a / 5)), a * 1.5
    else:
        r_ = float(s[4:]); d = (4 * V / (np.pi * r_)) ** (1 / 3); args, ext = ('wire', (d, d * r_)), d * r_ + 2
    n = int(np.ceil(ext + 2 * 3 * c.sigma + 6)); n += n % 2
    sim.resize(max(n, 12))
    L = sim.path_length(args[0], args[1], rot=rot, off=off, seed=int(rng.integers(1 << 30)))
    return sim.od_map(L, cond['mat'], c.p_mm, c.sigma), off


models = {} if os.environ.get('GVXR_NO_V3') else {f'v2/B_s{s}': (E.load_model(f'{K}/runs/v2/B_s{s}/weights/best.pt'), 416) for s in range(3)}
for v in VARS:
    models.update({r: (E.load_model(sp), 640) for r, sp in NB.specs(v).items()})
print('모델', len(models), flush=True)
rows, fps = [], []
for n_img, stem in enumerate(vt):
    m = split.machine[stem]
    edt, passes = sites_for(stem)
    bg = cv2.imread(f'{K}/data/synth_val/erased/images/{stem}.png', 0).astype(np.float32); H, W = bg.shape
    for axis, level, cond in COND:
        rng = np.random.default_rng(zlib.crc32(f'{stem}|{axis}|{level}'.encode()))
        for gi, sites in enumerate(passes):
            Gm = np.zeros((H, W), np.float32); feet, info = [], []
            for x, y in sites:
                od, off = object_od(cond, m, rng); n = od.shape[0]
                X0, Y0 = x - n // 2, y - n // 2     # 물체 중심 = 패치 좌표 n/2 - 0.5 + off → 영상 x - 0.5 + off
                ys, xs = slice(max(Y0, 0), min(Y0 + n, H)), slice(max(X0, 0), min(X0 + n, W))
                Gm[ys, xs] += od[ys.start - Y0:ys.stop - Y0, xs.start - X0:xs.stop - X0]
                f = np.zeros((H, W), bool); f[ys, xs] = od[ys.start - Y0:ys.stop - Y0, xs.start - X0:xs.stop - X0] > 0.05 * od.max()
                feet.append(cv2.dilate(f.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0)
                info.append(dict(cx=x - 0.5 + off[0], cy=y - 0.5 + off[1], od_peak=float(od.max()), area=int((od > 0.5 * od.max()).sum()),
                                 contrast=float(bg[min(y, H - 1), min(x, W - 1)] * (1 - np.exp(-od.max()))), edge_dist=float(edt[min(y, H - 1), min(x, W - 1)])))
            im = insert(bg, Gm, OD=1.0)
            for run, (mdl, sz) in models.items():
                _, det = E.predict_img(mdl, im, sz); d = det.numpy()
                c_ = (d[:, :2] + d[:, 2:4]) / 2 if len(d) else np.zeros((0, 2))
                hit_any = np.zeros(len(d), bool)
                for j, inf in enumerate(info):
                    if len(d):
                        ci = np.clip(c_.astype(int), 0, [W - 1, H - 1])
                        near = (np.hypot(c_[:, 0] - inf['cx'], c_[:, 1] - inf['cy']) <= 8) | feet[j][ci[:, 1], ci[:, 0]]
                    else:
                        near = np.zeros(0, bool)
                    hit_any |= near
                    rows.append(dict(axis=axis, level=level, model=run, stem=stem, machine=m, pass_=gi, site=j, **inf,
                                     conf=float(d[near, 4].max()) if near.any() else 0.0))
                fps.append(dict(axis=axis, level=level, model=run, stem=stem, pass_=gi, n_fp=int(((d[:, 4] >= .25) & ~hit_any).sum()) if len(d) else 0))
    if (n_img + 1) % 10 == 0:
        print(n_img + 1, flush=True)
TAG = os.environ.get('GVXR_TAG', '')   # 결과 파일 이름 구분 (예: 새로 학습한 변형만 평가할 때)
D = pd.DataFrame(rows); D.to_csv(f'{OUT}/stress_site{TAG}.csv', index=False)
pd.DataFrame(fps).to_csv(f'{OUT}/stress_fp{TAG}.csv', index=False)
D['hit'] = D.conf >= .25; D['fam'] = D.model.str.replace(r'_s\d$', '', regex=True)
pd.set_option('display.width', 250)
print(D.groupby(['axis', 'level', 'fam'], sort=False).hit.mean().unstack('fam').round(3).to_string())
