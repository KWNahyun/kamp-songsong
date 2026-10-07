"""스트레스 곡선 v0 — 실제 결함 자리(관측된 위치·배경)에서 대비만 낮춰 가며 검출 한계를 잰다 (학습 없음, val·test 150장, 베이스라인 seed 3개)
축 N (잡음): 실제 영상에 가우시안 잡음 σ_add 추가. 결함은 진짜 → 합성 결함 없이 '촬영 조건 악화'만
축 D (밀도): 실제 결함을 지운 배경(data/synth_val/erased)에 결함을 다시 넣되 광학 밀도를 배율 k 로 조절
     D_param : 매개변수 합성 (사각 물체 ⊗ 흐림, 시도 2 + 보정)
     D_trans : 옮겨 심기 (다른 날짜 train 결함의 실제 광학 밀도 맵)
     → 두 합성기의 곡선이 같은지도 확인 (합성 방법에 따라 결론이 달라지면 안 됨)
결함별로 실제 측정한 CNR(대비 / 잡음)을 기록해 'CNR 대비 검출률' 곡선과 CNR50(검출률 50% 지점)을 구한다
출력: eval/synth/stress_v0_per_defect.csv, stress_v0_fp.csv, eda/fig/s1_stress_cnr.png, 콘솔 요약
환경변수 NB2_MODEL=yolov8s|dfine_s: 새 베이스라인 seed 3개 (640) → stress_v0_per_defect_<모델>.csv, stress_v0_fp_<모델>.csv
"""
import os, sys
import numpy as np
import pandas as pd
import cv2
from scipy import ndimage as ndi

K = '/data/knhyun/KAMP'
sys.path.insert(0, f'{K}/eval'); sys.path.insert(0, f'{K}/prep/synth'); sys.argv = ['x']
import evaluate as E
from synth_core import box, insert

OUT = f'{K}/eval/synth'
NOISE = [0, 1, 2, 3, 4, 6, 8, 12]
SCALE = [0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0, 1.3]
split = pd.read_csv(f'{K}/data/splits/split.csv'); vt = split[split.split.isin(['val', 'test'])].stem.tolist()
P = pd.read_csv(f'{OUT}/replica_params.csv'); P = P[P.stem.isin(vt)]
kcal = {(r.machine, r.version): r.k for r in pd.read_csv(f'{OUT}/od_calibration.csv').itertuples()}
ODMAP = {tuple([k.split('|')[0], int(k.split('|')[1])]): v for k, v in np.load(f'{OUT}/od_maps.npy', allow_pickle=True).item().items()}
R_T = 7


def measure(img, px, py):
    cf = img.astype(np.float32); H, W = img.shape
    box2 = cv2.blur(cf, (2, 2), anchor=(0, 0), borderType=cv2.BORDER_REPLICATE)
    y0, y1, x0, x1 = max(py - 2, 0), min(py + 3, H), max(px - 2, 0), min(px + 3, W)
    my, mx = np.unravel_index(np.argmin(box2[y0:y1, x0:x1]), (y1 - y0, x1 - x0)); qy, qx = y0 + my, x0 + mx
    win = lambda a, r: a[max(qy - r, 0):qy + r + 2, max(qx - r, 0):qx + r + 2]
    w9 = win(cf, 4).copy(); cm = np.zeros_like(w9, bool); cy0, cx0 = min(qy, 4), min(qx, 4)
    cm[max(cy0 - 2, 0):cy0 + 4, max(cx0 - 2, 0):cx0 + 4] = True
    bg = float(np.median(w9[~cm]))
    resid = cf - cv2.medianBlur(img, 5).astype(np.float32)
    sig = float(np.median(np.abs(win(resid, 7))) * 1.4826)
    return bg - float(box2[qy, qx]), sig


def make(stem, axis, level, rng):
    p = P[P.stem == stem]
    if axis == 'N':
        im = cv2.imread(f'{K}/data/clean/images/{stem}.png', 0).astype(np.float32)
        return np.clip(np.round(im + rng.normal(0, level, im.shape)), 0, 255).astype(np.uint8) if level else im.astype(np.uint8)
    bg = cv2.imread(f'{K}/data/synth_val/erased/images/{stem}.png', 0).astype(np.float32); H, W = bg.shape
    G = np.zeros((H, W), np.float32)
    for r in p.itertuples():
        if axis == 'D_param':
            G += level * kcal.get((r.machine, 'R1_mult'), 1.0) * r.OD0 * box((H, W), r.x0, r.y0, r.w, r.sigma)
        else:
            m = ODMAP.get((r.donor_stem, r.donor_obj)); y0, x0 = r.py - R_T, r.px - R_T
            if m is not None and y0 >= 0 and x0 >= 0 and y0 + m.shape[0] <= H and x0 + m.shape[1] <= W:
                G[y0:y0 + m.shape[0], x0:x0 + m.shape[1]] += level * m
    return insert(bg, G, OD=1.0)


NB2 = os.environ.get('NB2_MODEL'); SIZE, TAG = 416, ''
if NB2:
    import nb2_models as NB
    NB.patch(E); SIZE, TAG = 640, f'_{NB2}'
    models = {r: E.load_model(sp) for r, sp in NB.specs(NB2).items()}
else:
    models = {f'v2/B_s{s}': E.load_model(f'{K}/runs/v2/B_s{s}/weights/best.pt') for s in range(3)}
rows, fps = [], []
for axis, levels in [('N', NOISE), ('D_param', SCALE), ('D_trans', SCALE)]:
    for lv in levels:
        rng = np.random.default_rng(1)
        for stem in vt:
            im = make(stem, axis, lv, rng); p = P[P.stem == stem]
            gts = np.c_[p.px + 1.0, p.py + 1.0]   # 2×2 코어 중심
            meas = [measure(im, r.px, r.py) for r in p.itertuples()]
            for run, m in models.items():
                _, det = E.predict_img(m, im, SIZE); d = det.numpy()
                c = (d[:, :2] + d[:, 2:4]) / 2 if len(d) else np.zeros((0, 2))
                for (r, (con, sig), g) in zip(p.itertuples(), meas, gts):
                    near = np.hypot(c[:, 0] - g[0], c[:, 1] - g[1]) <= 8 if len(d) else np.zeros(0, bool)
                    rows.append(dict(axis=axis, level=lv, model=run, stem=stem, obj=r.obj, machine=r.machine,
                                     contrast=con, noise=sig, cnr=con / max(sig, .5), conf=float(d[near, 4].max()) if near.any() else 0.0))
                hi = d[:, 4] >= 0.25 if len(d) else np.zeros(0, bool)
                far = (np.hypot(c[hi, None, 0] - gts[None, :, 0], c[hi, None, 1] - gts[None, :, 1]).min(1) > 8) if hi.any() and len(gts) else np.ones(hi.sum(), bool)
                fps.append(dict(axis=axis, level=lv, model=run, stem=stem, n_fp=int(far.sum())))
        print(axis, lv, 'done', flush=True)
D = pd.DataFrame(rows); D.to_csv(f'{OUT}/stress_v0_per_defect{TAG}.csv', index=False)
F = pd.DataFrame(fps); F.to_csv(f'{OUT}/stress_v0_fp{TAG}.csv', index=False)
pd.set_option('display.width', 220)
s = D.groupby(['axis', 'level', 'machine']).agg(cnr=('cnr', 'median'), recall25=('conf', lambda x: (x >= .25).mean()), recall10=('conf', lambda x: (x >= .1).mean())).round(3)
print(s.unstack('machine').to_string())
print(F.groupby(['axis', 'level']).n_fp.mean().unstack(0).round(3))
