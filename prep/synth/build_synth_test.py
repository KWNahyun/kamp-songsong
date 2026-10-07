"""합성 test 세트 내보내기 → handoff/kamp_synth_test_v1/ (+ .zip)
합성 데이터는 평가 전용(팀 결정 2026-10-06). 배경은 val·test 150장뿐이며 train 사진은 쓰지 않는다.
보고서 §4.5(위치 축)·§4.6(gVXR 크기·형상·재질 축)과 같은 자리·같은 난수로 영상을 만들어 파일로 저장한다.
  erased            : 실제 결함을 지운 영상 (결함 없는 제품 대용, 라벨 없음)
  pos_canon_k{k}    : 원래 결함 자리(띠 끝)에 옮겨 심은 결함 (기증 = replica_params 의 donor, §4.3 T_cross)
  pos_random_k{k}   : 제품 안 무작위 자리 3곳 × 4회에 옮겨 심은 결함 (기증 = 같은 장비 train 결함)
  gvxr_{축}_{수준}  : gVXR 이물, 무작위 자리 3곳 × 2회 (stress_gvxr.py 의 COND)
라벨: YOLO 형식(class 0). 옮겨 심기는 장비별 실제 GT 박스 중앙값 크기, gVXR 는 이물 영역(OD > 최대의 5%) 외접 사각형 + 2px (최소 = GT 중앙값).
objects.csv: 물체마다 중심·박스·평가 영역(이물 영역을 4px 넓힌 사각형)·조건·대비. images.csv: 영상마다 세트·조건.
"""
import os, sys, zlib, shutil, zipfile
import numpy as np
import pandas as pd
import cv2
from scipy import ndimage as ndi

K = os.environ.get('KAMP_ROOT', '/data/knhyun/KAMP')
sys.path.insert(0, f'{K}/prep/synth')
from synth_core import insert
import gvxr_core as G

OUT = f'{K}/handoff/kamp_synth_test_v1'
split = pd.read_csv(f'{K}/data/splits/split.csv').set_index('stem')
vt = split[split.split.isin(['val', 'test'])].index.tolist()
P = pd.read_csv(f'{K}/eval/synth/replica_params.csv')
ODMAP = {tuple([k.split('|')[0], int(k.split('|')[1])]): v for k, v in np.load(f'{K}/eval/synth/od_maps.npy', allow_pickle=True).item().items()}
C = pd.read_csv(f'{K}/eval/synth/gvxr/calibration.csv').set_index('machine')
R_T = 7
POS_LEVELS = [0.5, 1.0]
COND = [('size', f'x{k}', dict(shape='sphere', mat='SUS304', scale=k)) for k in (0.5, 0.75, 1, 1.5, 2, 3, 5)] + \
       [('shape', s, dict(shape=s, mat='SUS304', scale=1)) for s in ('sphere', 'cube', 'irregular', 'plate', 'wire10', 'wire20')] + \
       [('material', f'{m}_x{k}', dict(shape='sphere', mat=m, scale=k)) for m in ('SUS304', 'Al', 'glass', 'stone', 'bone', 'plastic') for k in (1, 2, 4, 8)]
MID = {'1호기': 'M1', '2호기': 'M2', '3호기': 'M3'}


def product_mask(clean):   # eda/extract.py 와 같음
    blur = cv2.GaussianBlur(clean, (9, 9), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(th)
    if n <= 1:
        return np.zeros_like(th, bool)
    return ndi.binary_fill_holes(lab == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))


def candidates(stem):
    clean = cv2.imread(f'{K}/data/clean/images/{stem}.png', 0); H, W = clean.shape
    pm = product_mask(clean); edt = cv2.distanceTransform(pm.astype(np.uint8), cv2.DIST_L2, 5)
    p = P[P.stem == stem]; real = np.c_[p.px + 1.0, p.py + 1.0]
    yy, xx = np.nonzero(edt >= 2)
    ok = (yy >= R_T) & (xx >= R_T) & (yy + R_T + 2 <= H) & (xx + R_T + 2 <= W)
    ok &= np.hypot(xx[:, None] + 1 - real[None, :, 0], yy[:, None] + 1 - real[None, :, 1]).min(1) >= 20
    return edt, p, xx[ok], yy[ok]


train_pool = {m: [k for k in ODMAP if k[0] in split.index and split.split[k[0]] == 'train' and split.machine[k[0]] == m] for m in MID}


def pos_sites(stem):
    """stress_pos.sites_for 와 같음 (STRESS_DONOR 기본값 = train 기증)"""
    rng = np.random.default_rng(zlib.crc32(stem.encode()))
    edt, p, xx, yy = candidates(stem)
    passes = []
    for _ in range(4):
        ch = []
        for i in rng.permutation(len(xx)):
            if all(np.hypot(xx[i] - a, yy[i] - b) >= 24 for a, b, _ in ch):
                pl = train_pool[split.machine[stem]]
                ch.append((int(xx[i]), int(yy[i]), pl[rng.integers(len(pl))]))
            if len(ch) == 3:
                break
        passes.append(ch)
    canon = [(int(r.px), int(r.py), (r.donor_stem, int(r.donor_obj))) for r in p.itertuples()]
    return edt, passes, canon


def gvxr_sites(stem):
    """stress_gvxr.sites_for 와 같음 (같은 난수 소비 순서)"""
    rng = np.random.default_rng(zlib.crc32(stem.encode()))
    edt, p, xx, yy = candidates(stem)
    passes = []
    for _ in range(4):
        ch = []
        for i in rng.permutation(len(xx)):
            if all(np.hypot(xx[i] - a, yy[i] - b) >= 24 for a, b in ch):
                ch.append((int(xx[i]) + 1, int(yy[i]) + 1)); rng.integers(1000)
            if len(ch) == 3:
                break
        passes.append(ch)
    return passes[:2]


sim = G.Sim(os_=8, n_px=16)


def object_od(cond, machine, rng):   # stress_gvxr.object_od 와 같음
    c = C.loc[machine]; D = c.D_px * cond['scale']; V = np.pi / 6 * c.D_px ** 3
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


# 장비별 GT 박스 중앙값 (train 공식 라벨) → 옮겨 심기 박스 크기, gVXR 최소 박스
gtwh = {m: [] for m in MID}
for stem in split[split.split == 'train'].index:
    W, H = cv2.imread(f'{K}/data/clean/images/{stem}.png', 0).shape[::-1]
    for l in open(f'{K}/data/clean/labels/{stem}.txt'):
        if l.strip():
            _, _, _, w, h = map(float, l.split()); gtwh[split.machine[stem]].append((w * W, h * H))
WMIN = {m: np.median(np.array(v), 0) for m, v in gtwh.items()}
print('장비별 GT 박스 중앙값', {m: v.round(1).tolist() for m, v in WMIN.items()}, flush=True)

shutil.rmtree(OUT, ignore_errors=True)
for sp in ['val', 'test']:
    os.makedirs(f'{OUT}/images/{sp}'); os.makedirs(f'{OUT}/labels/{sp}')
os.makedirs(f'{OUT}/tools')
imgs, objs = [], []


def save(stem, set_, axis, level, pass_, im, boxes):
    sp = split.split[stem]; iid = f'{stem}__{set_}' + (f'__p{pass_}' if pass_ is not None else '')
    H, W = im.shape
    cv2.imwrite(f'{OUT}/images/{sp}/{iid}.png', im)
    with open(f'{OUT}/labels/{sp}/{iid}.txt', 'w') as f:
        for b in boxes:
            f.write(f"0 {b['cx'] / W:.6f} {b['cy'] / H:.6f} {b['bw'] / W:.6f} {b['bh'] / H:.6f}\n")
    imgs.append(dict(image_id=iid, split=sp, set=set_, axis=axis, level=level, pass_=pass_, source_image=stem,
                     machine=MID[split.machine[stem]], width=W, height=H, n_objects=len(boxes)))
    for j, b in enumerate(boxes):
        objs.append(dict(image_id=iid, split=sp, set=set_, axis=axis, level=level, machine=MID[split.machine[stem]], obj=j,
                         x1=b['cx'] - b['bw'] / 2, y1=b['cy'] - b['bh'] / 2, x2=b['cx'] + b['bw'] / 2, y2=b['cy'] + b['bh'] / 2,
                         **{k: v for k, v in b.items() if k not in ('bw', 'bh')}))


for n_img, stem in enumerate(vt):
    m = split.machine[stem]; wm = WMIN[m]
    bg = cv2.imread(f'{K}/data/synth_val/erased/images/{stem}.png', 0).astype(np.float32); H, W = bg.shape
    save(stem, 'erased', 'none', 'none', None, bg.astype(np.uint8), [])
    # 위치 축 (옮겨 심기)
    edt, passes, canon = pos_sites(stem)
    for kind, groups in [('canon', [canon]), ('random', passes)]:
        for gi, sites in enumerate(groups):
            for k in POS_LEVELS:
                Gm = np.zeros((H, W), np.float32); boxes = []
                for x, y, d in sites:
                    od = ODMAP[d]; Gm[y - R_T:y - R_T + od.shape[0], x - R_T:x - R_T + od.shape[1]] += k * od
                    cx, cy = x + 1.0, y + 1.0
                    boxes.append(dict(cx=cx, cy=cy, bw=float(wm[0]), bh=float(wm[1]), ex1=cx - 8, ey1=cy - 8, ex2=cx + 8, ey2=cy + 8,
                                      edge_dist=float(edt[min(int(cy), H - 1), min(int(cx), W - 1)]), donor=f'{d[0]}|{d[1]}', k=k))
                save(stem, f'pos_{kind}_k{k}', 'position', f'{kind}_k{k}', gi if kind == 'random' else None, insert(bg, Gm, OD=1.0), boxes)
    # gVXR 크기·형상·재질 축
    gpasses = gvxr_sites(stem)
    for axis, level, cond in COND:
        rng = np.random.default_rng(zlib.crc32(f'{stem}|{axis}|{level}'.encode()))
        for gi, sites in enumerate(gpasses):
            Gm = np.zeros((H, W), np.float32); boxes = []
            for x, y in sites:
                od, off = object_od(cond, m, rng); n = od.shape[0]
                X0, Y0 = x - n // 2, y - n // 2
                ys, xs = slice(max(Y0, 0), min(Y0 + n, H)), slice(max(X0, 0), min(X0 + n, W))
                Gm[ys, xs] += od[ys.start - Y0:ys.stop - Y0, xs.start - X0:xs.stop - X0]
                fy, fx = np.nonzero(od > 0.05 * od.max())
                fx1, fx2, fy1, fy2 = X0 + fx.min(), X0 + fx.max(), Y0 + fy.min(), Y0 + fy.max()
                x1, x2 = max(fx1 - 2, 0), min(fx2 + 3, W); y1, y2 = max(fy1 - 2, 0), min(fy2 + 3, H)
                cx, cy = x - 0.5 + off[0], y - 0.5 + off[1]
                boxes.append(dict(cx=(x1 + x2) / 2, cy=(y1 + y2) / 2, bw=float(max(x2 - x1, wm[0])), bh=float(max(y2 - y1, wm[1])),
                                  obj_cx=cx, obj_cy=cy, ex1=fx1 - 4, ey1=fy1 - 4, ex2=fx2 + 5, ey2=fy2 + 5,
                                  od_peak=float(od.max()), area=int((od > 0.5 * od.max()).sum()),
                                  contrast=float(bg[min(y, H - 1), min(x, W - 1)] * (1 - np.exp(-od.max()))),
                                  edge_dist=float(edt[min(y, H - 1), min(x, W - 1)]),
                                  shape=cond['shape'], material=cond['mat'], scale=cond['scale']))
            save(stem, f'gvxr_{axis}_{level}', axis, level, gi, insert(bg, Gm, OD=1.0), boxes)
    if (n_img + 1) % 10 == 0:
        print(n_img + 1, len(imgs), flush=True)

I = pd.DataFrame(imgs); O = pd.DataFrame(objs)
# 옮겨 심기는 물체 중심 = 박스 중심
O['obj_cx'] = O.obj_cx.fillna(O.cx); O['obj_cy'] = O.obj_cy.fillna(O.cy)
I.to_csv(f'{OUT}/images.csv', index=False); O.to_csv(f'{OUT}/objects.csv', index=False)
open(f'{OUT}/data.yaml', 'w').write('# ultralytics YOLO 설정. path 를 이 폴더의 절대 경로로 바꾸세요. 합성 test 전용 (학습 금지)\n'
                                    'path: .\nval: images/val\ntest: images/test\nnames:\n  0: defect\n')
for f in ['eval_synth.py', 'README.md']:
    shutil.copyfile(f'{K}/prep/handoff_src/synth_test/{f}', f'{OUT}/{"" if f == "README.md" else "tools/"}{f}')
print(I.groupby(['split', 'axis']).size().unstack(0))
print('영상', len(I), '물체', len(O))
with zipfile.ZipFile(f'{K}/handoff/kamp_synth_test_v1.zip', 'w', zipfile.ZIP_STORED) as z:   # PNG 는 이미 압축됨
    for root, _, files in os.walk(OUT):
        for f in files:
            z.write(os.path.join(root, f), os.path.relpath(os.path.join(root, f), f'{K}/handoff'))
print('패키지:', OUT, '/ 압축:', f'{K}/handoff/kamp_synth_test_v1.zip')
