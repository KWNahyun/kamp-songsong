"""합성 타당성 검증 (V1~V4) — prep/synth/build_replica.py 의 복제 데이터 사용
모델 = 각 사진을 학습에 쓰지 않은 모델 (val·test: 베이스라인 v2/B_s0~2, train: 교차 검증 교사 v2/CV_f<k>)

미리 정한 통과 기준 (결과를 보기 전에 고정):
  V1 물리·화소 : 결함 대비, CNR, 최대 광학 밀도, 반치 면적 — 실제 대비 표준화 평균차 |SMD| < 0.2
  V2 통계      : 결함 주변 잡음 σ |SMD| < 0.2, 반경 방향 밝기 단면 차이 < 결함 간 편차의 20%
                 구별 분류기 AUC ≤ 0.60 (보조 지표. 양성 대조(sharp·nonoise)가 AUC ≥ 0.70 이어야 이 검사에 '구별할 힘'이 있다고 봄)
  V3 모델 반응 : 놓침률(신뢰도 < 0.25) 차이 ≤ 2%p (전체), 기준 구간(3호기 CNR 8~10)에서 실제와 합성의 95% 구간이 겹침
                 신뢰도 평균 차 |Δ| < 0.05, 검출 박스 중심 오차 차 < 0.5px
  V4 빈 합성   : null_sites 반응률(신뢰도 ≥ 0.1)이 같은 위치의 원본 대비 +0.5%p 이하, 그 이미지의 실제 결함 신뢰도 변화 |Δ| < 0.02
사용: python eval/synth/validate_replica.py [--infer | --add]   (--infer: 검출부터 다시 계산, --add: 검출 파일에 없는 버전만 추론해 덧붙임. GPU 필요)
gVXR 버전 (prep/synth/build_gvxr_replica.py): G_gen(장비별 보정만), G_fit(결함별 크기 맞춤), H_gvxr(공정 비교, H_real 과 짝)
출력: eval/synth/replica_dets.csv, replica_per_defect.csv, validation_summary.csv, 콘솔 요약
환경변수 NB2_MODEL=yolov8s|dfine_s: 새 베이스라인 seed 3개 (640), val·test 결함만 → eval/synth/nb2_<모델>/ (보정 계수·복제 데이터는 기존 것 사용)
"""
import os, sys, argparse
import numpy as np
import pandas as pd
import cv2

K = '/data/knhyun/KAMP'
ap = argparse.ArgumentParser(); ap.add_argument('--infer', action='store_true'); ap.add_argument('--add', action='store_true'); args = ap.parse_args()
sys.path.insert(0, f'{K}/eval'); sys.argv = ['x']
OUT = f'{K}/eval/synth'
VERS = ['real', 'erased', 'T_self', 'T_cross', 'R1_mult', 'R1_cal', 'R1_snap', 'R1_snapcal', 'H_snap', 'R1_sharp', 'R1_nonoise', 'R2_mult', 'R2_add', 'H_real', 'H_syn', 'null_sites', 'G_gen', 'G_fit', 'H_gvxr']
path = lambda v, s: f'{K}/data/clean/images/{s}.png' if v == 'real' else f'{K}/data/synth_val/{v}/images/{s}.png'
split = pd.read_csv(f'{K}/data/splits/split.csv'); cv = pd.read_csv(f'{K}/data/splits/cv3.csv')
JOBS = [(f'v2/B_s{s}', split[split.split.isin(['val', 'test'])].stem.tolist()) for s in range(3)] + \
       [(f'v2/CV_f{k}', cv[cv.fold == k].stem.tolist()) for k in range(3)]
NB2 = os.environ.get('NB2_MODEL'); SIZE, OUT_R = 416, OUT
spec_of = lambda run: f'{K}/runs/{run}/weights/best.pt'
if NB2:
    sys.path.insert(0, f'{K}/eval'); import nb2_models as NB
    _sp = NB.specs(NB2); SIZE, OUT_R = 640, f'{OUT}/nb2_{NB2}'; os.makedirs(OUT_R, exist_ok=True)
    JOBS = [(r, split[split.split.isin(['val', 'test'])].stem.tolist()) for r in _sp]; spec_of = _sp.get

# ── 1. 검출
_have = set(pd.read_csv(f'{OUT_R}/replica_dets.csv', usecols=['version']).version) if args.add and os.path.exists(f'{OUT_R}/replica_dets.csv') else set()
if args.infer or args.add or not os.path.exists(f'{OUT_R}/replica_dets.csv'):
    import evaluate as E
    if NB2:
        NB.patch(E)
    rows = []
    for run, stems in JOBS:
        m = E.load_model(spec_of(run))
        for v in [v for v in VERS if v not in _have]:
            for s in stems:
                _, det = E.predict_img(m, cv2.imread(path(v, s), 0), SIZE)
                rows += [dict(version=v, model=run, stem=s, x1=d[0], y1=d[1], x2=d[2], y2=d[3], conf=d[4]) for d in det.numpy()]
        print('검출', run)
    _new = pd.DataFrame(rows)
    if _have:   # 검출이 0개인 버전도 다시 추론하지 않도록, 기존 검출에 덧붙임
        _new = pd.concat([pd.read_csv(f'{OUT_R}/replica_dets.csv'), _new], ignore_index=True)
    _new.to_csv(f'{OUT_R}/replica_dets.csv', index=False)
det = pd.read_csv(f'{OUT_R}/replica_dets.csv')
P = pd.read_csv(f'{OUT}/replica_params.csv')
obj = pd.read_csv(f'{K}/eda/out/objects.csv')
P = P.merge(obj[['stem', 'obj', 'cx', 'cy', 'bw', 'bh', 'W', 'H', 'cnr']], on=['stem', 'obj'])
P['date'] = P.stem.map(split.set_index('stem').date)
P['gx'] = P.cx * P.W; P['gy'] = P.cy * P.H


# ── 2. 결함별 화소 측정 (eda/extract.py 와 같은 정의)
def measure(img, px, py):
    cf = img.astype(np.float32); H, W = img.shape
    box2 = cv2.blur(cf, (2, 2), anchor=(0, 0), borderType=cv2.BORDER_REPLICATE)
    y0, y1, x0, x1 = max(py - 2, 0), min(py + 3, H), max(px - 2, 0), min(px + 3, W)
    my, mx = np.unravel_index(np.argmin(box2[y0:y1, x0:x1]), (y1 - y0, x1 - x0)); qy, qx = y0 + my, x0 + mx
    core = float(box2[qy, qx])
    win = lambda a, r: a[max(qy - r, 0):qy + r + 2, max(qx - r, 0):qx + r + 2]
    w9 = win(cf, 4).copy(); cm = np.zeros_like(w9, bool); cy0, cx0 = min(qy, 4), min(qx, 4)
    cm[max(cy0 - 2, 0):cy0 + 4, max(cx0 - 2, 0):cx0 + 4] = True
    bg = float(np.median(w9[~cm]))
    resid = cf - cv2.medianBlur(img, 5).astype(np.float32)
    sig = float(np.median(np.abs(win(resid, 7))) * 1.4826)
    c = bg - core
    w7 = win(cf, 3); dark = w7 < (bg - c / 2)
    from scipy import ndimage as ndi
    lab, _ = ndi.label(dark); cid = lab[min(qy, 3), min(qx, 3)]
    fwhm = int((lab == cid).sum()) if cid else 0
    patch = cf[qy - 5:qy + 7, qx - 5:qx + 7]
    prof = None
    if patch.shape == (12, 12):
        yy, xx = np.mgrid[:12, :12]; rr = np.hypot(yy - 5.5, xx - 5.5)
        prof = [float((patch[(rr >= a) & (rr < a + 1)] / bg).mean()) for a in range(6)]
    return dict(bg=bg, contrast=c, noise=sig, cnr=c / max(sig, .5), od_peak=-np.log(max(core, 1) / max(bg, 1)), fwhm=fwhm,
                patch=patch if patch.shape == (12, 12) else None, prof=prof)


def resp(d, gx, gy, gt_box):
    """결함 중심 8px 안 검출의 최대 신뢰도, 그 박스의 중심 오차·IoU"""
    if not len(d):
        return 0.0, np.nan, 0.0
    c = np.c_[(d.x1 + d.x2) / 2, (d.y1 + d.y2) / 2]; dist = np.hypot(c[:, 0] - gx, c[:, 1] - gy); near = dist <= 8
    if not near.any():
        return 0.0, np.nan, 0.0
    k = np.where(near)[0][np.argmax(d.conf.values[near])]; b = d.iloc[k]
    ix = max(0, min(b.x2, gt_box[2]) - max(b.x1, gt_box[0])); iy = max(0, min(b.y2, gt_box[3]) - max(b.y1, gt_box[1]))
    inter = ix * iy; iou = inter / ((b.x2 - b.x1) * (b.y2 - b.y1) + (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1]) - inter)
    return float(b.conf), float(dist[k]), float(iou)


model_of = {}
for run, stems in JOBS:
    for s in stems:
        model_of.setdefault(s, []).append(run)
dg = {k: g for k, g in det.groupby(['version', 'model', 'stem'])}
rows, patches = [], []
for s, p in P[P.stem.isin(model_of)].groupby('stem'):
    imgs = {v: cv2.imread(path(v, s), 0) for v in VERS}
    for r in p.itertuples():
        gt_box = [r.gx - r.bw * r.W / 2, r.gy - r.bh * r.H / 2, r.gx + r.bw * r.W / 2, r.gy + r.bh * r.H / 2]
        for v in VERS:
            m = measure(imgs[v], r.px, r.py)
            if m['patch'] is not None and v != 'null_sites':
                patches.append(dict(version=v, stem=s, obj=r.obj, machine=r.machine, date=r.date, patch=(m['patch'] / max(m['bg'], 1)).ravel()))
            for run in model_of[s]:
                conf, off, iou = resp(dg.get((v, run, s), det.iloc[:0]), r.gx, r.gy, gt_box)
                rows.append(dict(version=v, model=run, stem=s, obj=r.obj, machine=r.machine, cnr_real=r.cnr, date=r.date,
                                 conf=conf, offset=off, iou=iou, **{k: m[k] for k in ['bg', 'contrast', 'noise', 'cnr', 'od_peak', 'fwhm']},
                                 **{f'prof{i}': (m['prof'][i] if m['prof'] else np.nan) for i in range(6)}))
D = pd.DataFrame(rows); D.to_csv(f'{OUT_R}/replica_per_defect.csv', index=False)
# OD 보정 계수: train 결함만으로 (실제 / R1_mult 의 평균 광학 밀도 비). 보정 결과는 val·test 결함으로 판정
_t = D[(D.stem.map(split.set_index('stem').split) == 'train')].drop_duplicates(['version', 'stem', 'obj'])
_old = pd.read_csv(f'{OUT}/od_calibration.csv') if os.path.exists(f'{OUT}/od_calibration.csv') else pd.DataFrame(columns=['machine', 'version', 'k'])
_new = []
for _v in ['R1_mult', 'R1_snap']:
    if (_old.version == _v).any():
        continue
    _k = _t[_t.version == 'real'].groupby('machine').od_peak.mean() / _t[_t.version == _v].groupby('machine').od_peak.mean()
    _new += [dict(machine=m, version=_v, k=k) for m, k in _k.items()]
if _new and not NB2:
    pd.concat([_old, pd.DataFrame(_new)]).to_csv(f'{OUT}/od_calibration.csv', index=False)
    print('OD 보정 계수 추가 (다시 build_replica.py 실행 필요):', _new)

# ── 3. 판정
pd.set_option('display.width', 250)
real = D[D.version == 'real'].set_index(['model', 'stem', 'obj'])
summ = []
smd = lambda a, b: (b.mean() - a.mean()) / np.sqrt((a.var() + b.var()) / 2)
one = D.drop_duplicates(['version', 'stem', 'obj'])   # 화소 지표는 모델과 무관
r1 = one[one.version == 'real']
EVAL_SPLIT = one.stem.map(split.set_index('stem').split).isin(['val', 'test'])   # 보정을 train 으로 했으므로 판정은 val·test 결함
for v in ['T_self', 'T_cross', 'R1_mult', 'R1_cal', 'R1_snap', 'R1_snapcal', 'H_syn', 'H_snap', 'R1_sharp', 'R1_nonoise', 'R2_mult', 'R2_add', 'G_gen', 'G_fit', 'H_gvxr']:
    x = one[(one.version == v) & EVAL_SPLIT]; r1 = one[(one.version == 'real') & EVAL_SPLIT]; row = dict(version=v)
    for k in ['contrast', 'cnr', 'od_peak', 'fwhm', 'noise']:
        row[f'SMD_{k}'] = smd(r1[k], x[k])
    pr = [f'prof{i}' for i in range(6)]
    row['profile_diff_rel'] = float(np.abs(x[pr].mean().values - r1[pr].mean().values).max() / r1[pr].std().mean())
    y = D[D.version == v].set_index(['model', 'stem', 'obj']).reindex(real.index)
    row['n_defect_pixel'] = len(x)
    row['miss_real'] = (real.conf < .25).mean(); row['miss_syn'] = (y.conf < .25).mean()
    row['dconf_mean'] = (y.conf - real.conf).mean(); row['conf_corr'] = np.corrcoef(real.conf, y.conf)[0, 1]
    row['agree_det'] = ((real.conf >= .25) == (y.conf >= .25)).mean()
    row['doffset'] = (y.offset - real.offset).mean(); row['iou_ok_real'] = (real.iou >= .5).mean(); row['iou_ok_syn'] = (y.iou >= .5).mean()
    a = (real.machine == '3호기') & (real.cnr_real > 8) & (real.cnr_real <= 10)
    vt = real.index.get_level_values('stem').map(split.set_index('stem').split).isin(['val', 'test'])
    row['miss_real_vt'] = (real.conf[vt] < .25).mean(); row['miss_syn_vt'] = (y.conf[vt] < .25).mean()
    row['anchor_n'] = int(a.sum()); row['anchor_miss_real'] = (real.conf[a] < .25).mean(); row['anchor_miss_syn'] = (y.conf[a] < .25).mean()
    summ.append(row)
S = pd.DataFrame(summ)

# 구별 분류기 (보조): 장비·날짜 묶음 단위 교차 검증
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
PA = pd.DataFrame(patches)
def feats(X):
    X = X.reshape(-1, 12, 12)
    gx = np.abs(np.diff(X, axis=2)).reshape(len(X), -1); gy = np.abs(np.diff(X, axis=1)).reshape(len(X), -1)
    lap = np.stack([cv2.Laplacian(x.astype(np.float32), cv2.CV_32F).ravel() for x in X])
    return np.hstack([X.reshape(len(X), -1), gx, gy, np.abs(lap)])
auc = {}
for key in ['T_self', 'T_cross', 'T_cross|T_self', 'R1_cal|T_self', 'R1_mult', 'R1_cal', 'R1_snapcal', 'H_syn', 'H_snap', 'R1_sharp', 'R1_nonoise', 'R2_mult', 'R2_add', 'G_gen', 'G_fit', 'G_gen|T_self', 'G_fit|T_self', 'H_gvxr', 'real_split']:
    v = key
    if v == 'real_split':   # 음성 대조: 실제 결함을 무작위로 둘로 나눔
        a = PA[PA.version == 'real'].copy(); lab = np.random.default_rng(0).integers(0, 2, len(a)); X = feats(np.stack(a.patch)); grp = (a.machine + a.date.astype(str)).values
    else:
        ref = 'H_real' if v in ('H_syn', 'H_snap', 'H_gvxr') else 'real'   # 공정 비교는 같은 지우기 배경의 실제 결함과
        if '|' in v:
            v, ref = v.split('|')   # 'A|B' = A 를 B 와 비교 (같은 처리 과정끼리)
        a = PA[PA.version.isin([ref, v])]; lab = (a.version == v).astype(int).values; X = feats(np.stack(a.patch)); grp = (a.machine + a.date.astype(str)).values
    for name, clf in [('logreg', LogisticRegression(C=0.1, max_iter=3000)), ('gbm', HistGradientBoostingClassifier(max_iter=200, learning_rate=.05))]:
        pred = np.zeros(len(lab))
        for tr, te in GroupKFold(5).split(X, lab, grp):
            clf.fit(X[tr], lab[tr]); pred[te] = clf.predict_proba(X[te])[:, 1]
        auc[(key, name)] = roc_auc_score(lab, pred)
A = pd.Series(auc).unstack(); A.to_csv(f'{OUT_R}/classifier_auc.csv')
S = S.merge(A.max(1).rename('AUC_max').reset_index().rename(columns={'index': 'version'}), on='version', how='left')

# V4 빈 합성
ns = pd.read_csv(f'{OUT}/null_sites.csv')
v4 = []
for r in ns[ns.stem.isin(model_of)].itertuples():
    for run in model_of[r.stem]:
        for v in ['real', 'null_sites']:
            d = dg.get((v, run, r.stem), det.iloc[:0])
            cmax = 0.0
            if len(d):
                c = np.c_[(d.x1 + d.x2) / 2, (d.y1 + d.y2) / 2]; near = np.hypot(c[:, 0] - r.x, c[:, 1] - r.y) <= 8
                cmax = float(d.conf.values[near].max()) if near.any() else 0.0
            v4.append(dict(version=v, model=run, stem=r.stem, x=r.x, y=r.y, conf=cmax))
V4 = pd.DataFrame(v4)
er = D[D.version == 'erased']; nsd = D[D.version == 'null_sites'].set_index(['model', 'stem', 'obj']).reindex(real.index)
v4s = dict(null_ctrl_hit10=(V4[V4.version == 'real'].conf >= .1).mean(), null_hit10=(V4[V4.version == 'null_sites'].conf >= .1).mean(),
           null_hit25=(V4[V4.version == 'null_sites'].conf >= .25).mean(), null_sites=len(ns),
           null_img_real_defect_dconf=float((nsd.conf - real.conf).abs().mean()),
           erased_site_hit10=(er.conf >= .1).mean(), erased_site_hit25=(er.conf >= .25).mean(), erased_conf_mean=er.conf.mean())
S.to_csv(f'{OUT_R}/validation_summary.csv', index=False); pd.Series(v4s).to_csv(f'{OUT_R}/validation_v4.csv')
print(S.round(3).T.to_string()); print(A.round(3)); print(pd.Series(v4s).round(4))
