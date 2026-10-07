"""합성 타당성 검증용 복제 데이터 (GT 500장, 공식 라벨 그대로) — 학습에 쓰지 않음
각 GT 이미지에서 실제 결함을 모두 지우고(Telea + 잡음 복원), 같은 자리에 합성 결함을 다시 넣는다.
버전 (data/synth_val/<버전>/images/<stem>.png):
  erased     : 실제 결함을 지우고 잡음만 복원 (결함 없음)                          → V4 빈 합성 대조
  T_self     : 자기 결함의 실제 광학 밀도 맵을 같은 배경(지우고 잡음 복원)에 다시 곱함 → 처리 과정만 같고 결함은 진짜 (공정 비교 기준)
  T_cross    : 다른 날짜(train, 같은 장비) 결함의 실제 광학 밀도 맵을 옮겨 심음 (TIP 방식) → 옮겨 심기 검증
  R1_mult    : 결함마다 맞춘 OD0·크기 w·흐림 σ·위치로 곱셈 합성 (제안 방법)                   → V1~V3 짝 비교
  R1_cal     : R1 에 train 결함으로 정한 장비별 OD 보정 계수를 곱함 (eval/synth/od_calibration.csv)
  R1_snap / R1_snapcal : 중심을 픽셀 경계에 고정하고 다시 맞춤 (+ 보정)
  H_real / H_syn / H_snap : 공정 비교. 지운 영역 배경은 같고 결함 픽셀만 실제 / 합성(R1_cal) → 분류기가 '지우기 흔적' 없이 결함 모양만 비교
  R1_sharp   : R1 과 같은 최대 OD 이지만 흐림 없는 2×2 사각 점                      → 양성 대조 (틀린 모양)
  R1_nonoise : R1 과 같지만 지운 영역 잡음 복원 안 함                                → 양성 대조 (틀린 잡음)
  R2_mult    : 장비별 중앙값 OD0·w·σ 로 곱셈 합성 (결함별 조정 없이 '생성 모델'만으로)  → 생성 모델 검증
  R2_add     : 장비별 중앙값 밝기 차로 덧셈 합성                                     → 곱셈 vs 덧셈 비교
  null_sites : 결함 없는 제품 내부 3곳에 같은 크기 영역을 지우고 잡음만 복원 (실제 결함은 그대로) → V4
출력: data/synth_val/..., eval/synth/replica_params.csv (결함별 맞춤 값), eval/synth/null_sites.csv
"""
import os, sys
import numpy as np
import pandas as pd
import cv2

K = '/data/knhyun/KAMP'
sys.path.insert(0, f'{K}/prep/synth'); sys.argv = ['x']
exec(open(f'{K}/eda/extract.py').read().split('# ─────────────────────────── 1.')[0])   # product_mask
from synth_core import core_mask, erase, refill_noise, box, square2, fit_box, insert, KER3

OUT = f'{K}/data/synth_val'
VERS = ['erased', 'T_self', 'T_cross', 'R1_mult', 'R1_cal', 'R1_snap', 'R1_snapcal', 'H_snap', 'R1_sharp', 'R1_nonoise', 'R2_mult', 'R2_add', 'H_real', 'H_syn', 'null_sites']
CAL = f'{K}/eval/synth/od_calibration.csv'   # train 결함으로만 정한 장비별 OD 보정 계수 (validate_replica.py 가 작성)
_c = pd.read_csv(CAL) if os.path.exists(CAL) else pd.DataFrame(columns=['machine', 'version', 'k'])
kcal = {(r.machine, r.version): r.k for r in _c.itertuples()}
for v in VERS:
    os.makedirs(f'{OUT}/{v}/images', exist_ok=True)
os.makedirs(f'{K}/eval/synth', exist_ok=True)
obj = pd.read_csv(f'{K}/eda/out/objects.csv')
rng = np.random.default_rng(0)

# 1) 지우기 + 결함별 맞춤
params, cache = [], {}
for s, g in obj.groupby('stem'):
    img = cv2.imread(f'{K}/data/clean/images/{s}.png', 0)
    masks, info = [], []
    for r in g.itertuples():
        px = int(round(r.cx * r.W + r.dark_dx - 1)); py = int(round(r.cy * r.H + r.dark_dy - 1))
        m = core_mask(img, px, py, r.bg_local, r.contrast)
        masks.append(m); info.append((r, px, py))
    union = np.clip(np.sum(masks, 0), 0, 1).astype(np.uint8)
    er = erase(img, union)
    bgf = er.astype(np.float32)
    for m, (r, px, py) in zip(masks, info):
        bgf[m > 0] = refill_noise(er, m, r.noise_local, rng)[m > 0]
    for m, (r, px, py) in zip(masks, info):
        f = fit_box(img, er, px, py)
        params.append(dict(stem=s, obj=r.obj, machine=r.machine, px=px, py=py, mask_px=int(m.sum()), bg_local=r.bg_local,
                           contrast=r.contrast, noise=r.noise_local, **f))
    cache[s] = (img, er, bgf, union)
P = pd.DataFrame(params)
med = P.groupby('machine').agg(OD0_med=('OD0', 'median'), w_med=('w', 'median'), s_med=('sigma', 'median'), ODp_med=('OD_peak', 'median'))
P = P.join(med, on='machine')
P['C_med'] = P.machine.map(P.assign(c=P.bg_local * (1 - np.exp(-P.ODp_med))).groupby('machine').c.median())   # 덧셈형: 최대 밝기 차 고정
# 옮겨 심기(TIP)용 실제 결함 광학 밀도 맵: -ln(실제/지운 배경), 결함 자리(맞춤 상자 > 최대의 10%)를 1px 넓힌 영역만 (주변 잡음은 버림)
R_T = 7
def od_patch(s, r):
    img, er = cache[s][0], cache[s][1]; H, W = img.shape
    y0, x0 = r.py - R_T, r.px - R_T
    if y0 < 0 or x0 < 0 or y0 + 2 * R_T + 2 > H or x0 + 2 * R_T + 2 > W:
        return None
    sl = np.s_[y0:y0 + 2 * R_T + 2, x0:x0 + 2 * R_T + 2]
    od = -np.log(np.clip(img[sl].astype(np.float32) / np.maximum(er[sl].astype(np.float32), 1), 1e-3, 1.5))
    b = box(od.shape, r.x0 - x0, r.y0 - y0, r.w, r.sigma)
    fm = cv2.dilate((b > 0.1 * b.max()).astype(np.uint8), KER3, iterations=1) > 0
    return np.where(fm, np.clip(od, 0, None), 0).astype(np.float32)
ODMAP = {(r.stem, r.obj): od_patch(r.stem, r) for r in P.itertuples()}
np.save(f'{K}/eval/synth/od_maps.npy', {f'{k[0]}|{k[1]}': v for k, v in ODMAP.items() if v is not None}, allow_pickle=True)
# 기증 결함: train split, 같은 장비, 다른 (장비, 날짜) 묶음에서 무작위 1개 (seed 0)
spl = pd.read_csv(f'{K}/data/splits/split.csv').set_index('stem')
P['group'] = P.stem.map(spl.group); P['split'] = P.stem.map(spl.split)
pool = P[(P.split == 'train') & P.apply(lambda r: ODMAP[(r.stem, r.obj)] is not None, axis=1)]
drng = np.random.default_rng(0)
don = []
for r in P.itertuples():
    c = pool[(pool.machine == r.machine) & (pool.group != r.group)]
    d = c.iloc[drng.integers(len(c))]
    don.append((d.stem, d.obj))
P['donor_stem'] = [d[0] for d in don]; P['donor_obj'] = [d[1] for d in don]
P.to_csv(f'{K}/eval/synth/replica_params.csv', index=False)

# 2) 버전별 이미지
null_rows = []
for s, (img, er, bgf, union) in cache.items():
    p = P[P.stem == s]; H, W = img.shape
    G1 = np.zeros((H, W), np.float32); Gs = np.zeros_like(G1); G2 = np.zeros_like(G1); Ga = np.zeros_like(G1); Gc = np.zeros_like(G1)
    GTs = np.zeros_like(G1); GTc = np.zeros_like(G1)
    foot = np.zeros((H, W), bool); Gn = np.zeros_like(G1); Gnc = np.zeros_like(G1); foot_n = np.zeros((H, W), bool)
    for r in p.itertuples():
        b1 = box((H, W), r.x0, r.y0, r.w, r.sigma)
        G1 += r.OD0 * b1; Gc += kcal.get((r.machine, 'R1_mult'), 1.0) * r.OD0 * b1
        bn = box((H, W), r.sn_x0, r.sn_y0, r.sn_w, r.sn_sigma)
        Gn += r.sn_OD0 * bn; Gnc += kcal.get((r.machine, 'R1_snap'), 1.0) * r.sn_OD0 * bn; foot_n |= bn > 0.1 * bn.max()
        foot |= b1 > 0.1 * b1.max()   # 결함이 차지하는 픽셀 (공정 비교용)
        y0, x0 = r.py - R_T, r.px - R_T   # 옮겨 심기: 2×2 코어 기준점(px, py)을 맞춰 붙임
        for Gt, key in ((GTs, (r.stem, r.obj)), (GTc, (r.donor_stem, r.donor_obj))):
            m_ = ODMAP.get(key)
            if m_ is not None and y0 >= 0 and x0 >= 0 and y0 + m_.shape[0] <= H and x0 + m_.shape[1] <= W:
                Gt[y0:y0 + m_.shape[0], x0:x0 + m_.shape[1]] += m_
        Gs = np.maximum(Gs, r.OD_peak * square2((H, W), r.x0 - .5, r.y0 - .5))   # 흐림 없는 2×2 점: 같은 최대 OD
        g2 = box((H, W), r.x0, r.y0, r.w_med, r.s_med)
        G2 += r.OD0_med * g2; Ga += r.C_med * g2 / g2.max()
    out = {
        'erased': np.clip(np.round(bgf), 0, 255).astype(np.uint8),
        'T_self': insert(bgf, GTs, OD=1.0),
        'T_cross': insert(bgf, GTc, OD=1.0),
        'R1_mult': insert(bgf, G1, OD=1.0),
        'R1_cal': insert(bgf, Gc, OD=1.0),
        'R1_snap': insert(bgf, Gn, OD=1.0),
        'R1_snapcal': insert(bgf, Gnc, OD=1.0),
        'R1_sharp': insert(bgf, Gs, OD=1.0),
        'R1_nonoise': insert(er.astype(np.float32), G1, OD=1.0),
        'R2_mult': insert(bgf, G2, OD=1.0),
        'R2_add': insert(bgf, Ga, C=1.0, mode='add'),
    }
    # 공정 비교: 지운 영역은 둘 다 같은 배경(bgf), 결함 픽셀만 실제 / 합성(보정)
    hr = np.clip(np.round(bgf), 0, 255).astype(np.uint8); hr[foot] = img[foot]
    hs = np.clip(np.round(bgf), 0, 255).astype(np.uint8); hs[foot] = out['R1_cal'][foot]
    hn = np.clip(np.round(bgf), 0, 255).astype(np.uint8); hn[foot] = out['R1_snapcal'][foot]   # H_real 과 같은 결함 픽셀 영역
    out['H_real'], out['H_syn'], out['H_snap'] = hr, hs, hn
    # null_sites: 결함 없는 제품 내부 3곳 (실제 결함·서로 30px 이상), 지우기 영역은 이 이미지 결함 마스크 모양 재사용
    pm = product_mask(img); ys, xs = np.where(cv2.erode(pm.astype(np.uint8), np.ones((25, 25), np.uint8)) > 0)
    gts = p[['px', 'py']].values.astype(float)
    ny = img.copy().astype(np.float32); sites = []
    mshape = cv2.dilate(np.pad(np.ones((2, 2), np.uint8), 3), np.ones((3, 3), np.uint8), iterations=2)   # 2×2 코어 + 2px 확장
    for _ in range(2000):
        if len(sites) == 3 or not len(xs): break
        i = rng.integers(len(xs)); x, y = int(xs[i]), int(ys[i])
        if len(gts) and np.hypot(gts[:, 0] - x, gts[:, 1] - y).min() < 30: continue
        if any(np.hypot(a - x, b - y) < 30 for a, b in sites): continue
        sites.append((x, y))
    m = np.zeros((H, W), np.uint8)
    for x, y in sites:
        h = mshape.shape[0] // 2; m[y - h:y - h + mshape.shape[0], x - h:x - h + mshape.shape[1]] |= mshape
    if sites:
        e2 = erase(img, m); sig = float(p.noise.median())
        ny = img.astype(np.float32); ny[m > 0] = refill_noise(e2, m, sig, rng)[m > 0]
    out['null_sites'] = np.clip(np.round(ny), 0, 255).astype(np.uint8)
    for x, y in sites:
        null_rows.append(dict(stem=s, x=x, y=y))
    for v, im in out.items():
        cv2.imwrite(f'{OUT}/{v}/images/{s}.png', im)
pd.DataFrame(null_rows).to_csv(f'{K}/eval/synth/null_sites.csv', index=False)
print(P.groupby('machine')[['OD0', 'w', 'sigma', 'OD_peak', 'OD_obs_peak', 'fit_rmse']].describe(percentiles=[.05, .5, .95]).round(3).T.to_string())
print(med.round(3)); print('null sites', len(null_rows))
