"""스트레스: 위치 축 — 결함을 '늘 있던 자리(띠 끝)'가 아닌 제품 안 무작위 자리에 옮겨 심어, 위치 지름길 없이도 찾는지 잰다 (학습 없음)
배경: 실제 결함을 지운 val·test 150장 (data/synth_val/erased). 원래 결함 자리의 위치 반응이 섞이지 않게 함
결함: 옮겨 심기(TIP) — 같은 장비 train 결함의 실제 광학 밀도 맵(eval/synth/od_maps.npy)을 배율 k 로 곱셈 삽입 (§4.3 조건부 통과 방식)
  random : 제품 안 무작위 자리 3곳 × 4회 (원래 결함 자리와 20px 이상, 서로 24px 이상, 제품 가장자리에서 2px 이상). 기증 결함도 무작위
  canon  : 원래 결함 자리에 같은 방식으로 다시 심음 (기증 결함 = replica_params 의 donor, §4.3 T_cross 와 같음) → 위치 효과의 기준
  k = 0 : 아무것도 심지 않은 같은 자리의 반응 (자리 자체의 반응)
검출 = 심은 결함 중심 8px 안의 최대 신뢰도 (≥ 0.25 이면 찾음). 오검출 = 신뢰도 ≥ 0.25 인데 심은 자리 8px 밖 → 원래 결함 자리(지운 자리) / 그 밖
자리 특성: 제품 가장자리 거리, 지운 배경의 국소 밝기와 제품 안 밝기 순위, 국소 구조(9×9 표준편차), 원래 결함 자리까지 거리, 측정 대비
출력: eval/synth/stress_pos_site{TAG}.csv, stress_pos_fp{TAG}.csv
환경변수 STRESS_DONOR=heldout: 기증 결함을 train 대신 val·test 결함(같은 장비, 다른 (장비, 날짜) 묶음)으로 → 옮겨 심기 학습 모델이 학습 때 본 결함 맵과 겹치지 않게. 출력 이름 끝에 _hd
환경변수 NB2_MODEL=<모델 이름>: 새 베이스라인 seed 3개 (640, eval/nb2_models.specs). 없으면 YOLOv3 베이스라인 v2/B_s0~2 (416)
"""
import os, sys, zlib
import numpy as np
import pandas as pd
import cv2
from scipy import ndimage as ndi

K = '/data/knhyun/KAMP'
sys.path.insert(0, f'{K}/eval'); sys.path.insert(0, f'{K}/prep/synth'); sys.argv = ['x']
import evaluate as E
from synth_core import insert
OUT = f'{K}/eval/synth'
LEVELS = [0.0, 0.2, 0.35, 0.5, 0.75, 1.0]
NPASS, NSITE, R_T = 4, 3, 7


def product_mask(clean):   # eda/extract.py 와 같음
    blur = cv2.GaussianBlur(clean, (9, 9), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(th)
    if n <= 1:
        return np.zeros_like(th, bool)
    return ndi.binary_fill_holes(lab == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))


def measure(img, px, py):   # stress_v0.measure 의 대비 부분
    cf = img.astype(np.float32); H, W = img.shape
    box2 = cv2.blur(cf, (2, 2), anchor=(0, 0), borderType=cv2.BORDER_REPLICATE)
    y0, y1, x0, x1 = max(py - 2, 0), min(py + 3, H), max(px - 2, 0), min(px + 3, W)
    my, mx = np.unravel_index(np.argmin(box2[y0:y1, x0:x1]), (y1 - y0, x1 - x0)); qy, qx = y0 + my, x0 + mx
    w9 = cf[max(qy - 4, 0):qy + 6, max(qx - 4, 0):qx + 6].copy(); cm = np.zeros_like(w9, bool); cy0, cx0 = min(qy, 4), min(qx, 4)
    cm[max(cy0 - 2, 0):cy0 + 4, max(cx0 - 2, 0):cx0 + 4] = True
    return float(np.median(w9[~cm])) - float(box2[qy, qx])


split = pd.read_csv(f'{K}/data/splits/split.csv').set_index('stem')
vt = split[split.split.isin(['val', 'test'])].index.tolist()
P = pd.read_csv(f'{OUT}/replica_params.csv')
ODMAP = {tuple([k.split('|')[0], int(k.split('|')[1])]): v for k, v in np.load(f'{OUT}/od_maps.npy', allow_pickle=True).item().items()}
HD = os.environ.get('STRESS_DONOR') == 'heldout'
_sp = ['val', 'test'] if HD else ['train']
pool = {m: [k for k in ODMAP if k[0] in split.index and split.split[k[0]] in _sp and split.machine[k[0]] == m] for m in ['1호기', '2호기', '3호기']}


def sites_for(stem):
    """이미지별 고정 난수로 자리·기증 결함을 정함 (모델과 무관하게 같은 영상)"""
    rng = np.random.default_rng(zlib.crc32(stem.encode()))
    clean = cv2.imread(f'{K}/data/clean/images/{stem}.png', 0); H, W = clean.shape
    pm = product_mask(clean); edt = cv2.distanceTransform(pm.astype(np.uint8), cv2.DIST_L2, 5)
    p = P[P.stem == stem]; real = np.c_[p.px + 1.0, p.py + 1.0]
    yy, xx = np.nonzero(edt >= 2)
    ok = (yy >= R_T) & (xx >= R_T) & (yy + R_T + 2 <= H) & (xx + R_T + 2 <= W)
    ok &= np.hypot(xx[:, None] + 1 - real[None, :, 0], yy[:, None] + 1 - real[None, :, 1]).min(1) >= 20
    yy, xx = yy[ok], xx[ok]
    passes = []
    for _ in range(NPASS):
        ch = []
        for i in rng.permutation(len(xx)):
            if all(np.hypot(xx[i] - a, yy[i] - b) >= 24 for a, b, _ in ch):
                pl = [k for k in pool[split.machine[stem]] if split.group[k[0]] != split.group[stem]] if HD else pool[split.machine[stem]]
                ch.append((int(xx[i]), int(yy[i]), pl[rng.integers(len(pl))]))
            if len(ch) == NSITE:
                break
        passes.append(ch)
    if HD:   # 원래 자리에도 val·test 기증 결함 (다른 묶음, 같은 장비)
        pl = [k for k in pool[split.machine[stem]] if split.group[k[0]] != split.group[stem]]
        canon = [(int(r.px), int(r.py), pl[rng.integers(len(pl))]) for r in p.itertuples()]
    else:
        canon = [(int(r.px), int(r.py), (r.donor_stem, int(r.donor_obj))) for r in p.itertuples()]
    return pm, edt, real, passes, canon


def render(bg, sites, k):
    H, W = bg.shape; G = np.zeros((H, W), np.float32)
    for x, y, d in sites:
        m = ODMAP[d]; G[y - R_T:y - R_T + m.shape[0], x - R_T:x - R_T + m.shape[1]] += k * m
    return insert(bg, G, OD=1.0)


NB2 = os.environ.get('NB2_MODEL'); SIZE, TAG = 416, ''
if NB2:
    import nb2_models as NB
    NB.patch(E); SIZE, TAG = 640, f'_{NB2}'
    models = {r: E.load_model(sp) for r, sp in NB.specs(NB2).items()}
else:
    models = {f'v2/B_s{s}': E.load_model(f'{K}/runs/v2/B_s{s}/weights/best.pt') for s in range(3)}
TAG += '_hd' if HD else ''

rows, fps = [], []
for n_img, stem in enumerate(vt):
    pm, edt, real, passes, canon = sites_for(stem)
    bg = cv2.imread(f'{K}/data/synth_val/erased/images/{stem}.png', 0).astype(np.float32)
    pv = np.sort(bg[pm])
    loc_mean = cv2.blur(bg, (9, 9)); loc_sd = np.sqrt(np.maximum(cv2.blur(bg ** 2, (9, 9)) - loc_mean ** 2, 0))
    for kind, groups in [('random', passes), ('canon', [canon])]:
        for gi, sites in enumerate(groups):
            for k in LEVELS:
                im = render(bg, sites, k)
                ctr = np.array([[x + 1.0, y + 1.0] for x, y, _ in sites])
                for run, m in models.items():
                    _, det = E.predict_img(m, im, SIZE); d = det.numpy()
                    c = (d[:, :2] + d[:, 2:4]) / 2 if len(d) else np.zeros((0, 2))
                    for j, (x, y, dn) in enumerate(sites):
                        near = np.hypot(c[:, 0] - ctr[j, 0], c[:, 1] - ctr[j, 1]) <= 8 if len(d) else np.zeros(0, bool)
                        rows.append(dict(kind=kind, pass_=gi, level=k, model=run, stem=stem, machine=split.machine[stem], site=j, x=x, y=y,
                                         donor=f'{dn[0]}|{dn[1]}', conf=float(d[near, 4].max()) if near.any() else 0.0))
                    hi = d[:, 4] >= 0.25 if len(d) else np.zeros(0, bool)
                    ch = c[hi]
                    far = np.hypot(ch[:, None, 0] - ctr[None, :, 0], ch[:, None, 1] - ctr[None, :, 1]).min(1) > 8 if len(ch) else np.zeros(0, bool)
                    at_real = np.hypot(ch[:, None, 0] - real[None, :, 0], ch[:, None, 1] - real[None, :, 1]).min(1) <= 8 if len(ch) and len(real) else np.zeros(len(ch), bool)
                    fps.append(dict(kind=kind, pass_=gi, level=k, model=run, stem=stem, machine=split.machine[stem],
                                    fp_at_erased=int((far & at_real).sum()) if kind == 'random' else 0, fp_other=int((far & ~at_real).sum())))
    # 자리 특성 (모델 무관): 한 번만 계산해 붙임
    for kind, groups in [('random', passes), ('canon', [canon])]:
        for gi, sites in enumerate(groups):
            for j, (x, y, dn) in enumerate(sites):
                cx, cy = x + 1, y + 1
                feat = dict(edge_dist=float(edt[min(cy, edt.shape[0] - 1), min(cx, edt.shape[1] - 1)]), bg_local=float(loc_mean[cy, cx]),
                            bg_rank=float(np.searchsorted(pv, loc_mean[cy, cx]) / max(len(pv), 1)), bg_sd9=float(loc_sd[cy, cx]),
                            dist_real=float(np.hypot(real[:, 0] - cx, real[:, 1] - cy).min()) if len(real) else np.nan,
                            contrast1=measure(render(bg, [(x, y, dn)], 1.0), x, y))
                for r in rows[::-1]:
                    if r['stem'] != stem:
                        break
                    if r['kind'] == kind and r['pass_'] == gi and r['site'] == j:
                        r.update(feat)
    print(n_img + 1, stem, flush=True) if (n_img + 1) % 25 == 0 else None

D = pd.DataFrame(rows); D.to_csv(f'{OUT}/stress_pos_site{TAG}.csv', index=False)
F = pd.DataFrame(fps); F.to_csv(f'{OUT}/stress_pos_fp{TAG}.csv', index=False)
pd.set_option('display.width', 220)
D['hit'] = D.conf >= .25
print(D.groupby(['kind', 'level', 'machine']).hit.mean().unstack('machine').round(3).to_string())
print('자리 수', D[(D.level == 1) & (D.kind == 'random')].groupby('model').size().to_dict())
print(F.groupby(['kind', 'level'])[['fp_at_erased', 'fp_other']].mean().round(3).to_string())
