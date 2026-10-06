"""실패 조건 분석: 모든 GT 결함(1,147개)에 대해 '그 사진을 학습에 안 쓴 모델'의 반응을 보고, 어떤 조건에서 놓치는지 본다
입력: eval/failure/dets_heldout.csv (collect_heldout.py), eda/out/objects.csv (결함별 특성)
결함별 지표 (모델마다):
  conf_at : 결함 중심 8px 안 검출의 최대 신뢰도 (없으면 0)
  miss25  : conf_at < 0.25  (신뢰도 0.25 기준으로 놓침)
  iou_ok  : 그 최대 신뢰도 검출의 IoU ≥ 0.5 (위치는 맞췄는데 박스가 틀린 경우 구분)
오검출: conf ≥ 0.25 인데 어떤 GT 중심과도 8px 넘게 떨어진 검출
출력: eval/failure/per_defect.csv, per_fp.csv, 콘솔 요약
환경변수 NB2_MODEL=yolov8s|dfine_s: dets_heldout_<모델>.csv → per_defect_<모델>.csv, per_fp_<모델>.csv (val·test 결함 350개만)
"""
import os, sys
import numpy as np
import pandas as pd
K = '/data/knhyun/KAMP'
sys.path.insert(0, f'{K}/eval'); sys.argv = ['x']
exec(open(f'{K}/eda/extract.py').read().split('# ─────────────────────────── 1.')[0])   # product_mask
import cv2
MATCH, T = 8.0, 0.25
TAG = f"_{os.environ['NB2_MODEL']}" if os.environ.get('NB2_MODEL') else ''

det = pd.read_csv(f'{K}/eval/failure/dets_heldout{TAG}.csv')
obj = pd.read_csv(f'{K}/eda/out/objects.csv')
img = pd.read_csv(f'{K}/eda/out/images_with_session.csv').set_index('stem')
split = pd.read_csv(f'{K}/data/splits/split.csv').set_index('stem')
obj['gx'] = obj.cx * obj.W; obj['gy'] = obj.cy * obj.H
obj['gw'] = obj.bw * obj.W; obj['gh'] = obj.bh * obj.H


def iou1(b, g):
    ix = max(0, min(b[2], g[2]) - max(b[0], g[0])); iy = max(0, min(b[3], g[3]) - max(b[1], g[1]))
    inter = ix * iy
    return inter / ((b[2] - b[0]) * (b[3] - b[1]) + (g[2] - g[0]) * (g[3] - g[1]) - inter)


rows, fps = [], []
for (model, stem), d in det.groupby(['model', 'stem']):
    o = obj[obj.stem == stem]
    b = d[['x1', 'y1', 'x2', 'y2', 'conf']].values
    c = (b[:, :2] + b[:, 2:4]) / 2
    g = o[['gx', 'gy']].values
    for r, (gx, gy) in zip(o.itertuples(), g):
        dist = np.hypot(c[:, 0] - gx, c[:, 1] - gy); near = dist <= MATCH
        if near.any():
            k = np.where(near)[0][np.argmax(b[near, 4])]
            conf, iou = b[k, 4], iou1(b[k], [r.gx - r.gw / 2, r.gy - r.gh / 2, r.gx + r.gw / 2, r.gy + r.gh / 2])
            pw, ph = b[k, 2] - b[k, 0], b[k, 3] - b[k, 1]
        else:
            conf, iou, pw, ph = 0.0, 0.0, np.nan, np.nan
        rows.append(dict(model=model, stem=stem, obj=r.obj, conf_at=conf, iou=iou, pred_w=pw, pred_h=ph))
    hi = b[:, 4] >= T
    if hi.any():
        dmin = np.hypot(c[hi, None, 0] - g[None, :, 0], c[hi, None, 1] - g[None, :, 1]).min(1) if len(g) else np.full(hi.sum(), np.inf)
        for bb, cc, dm in zip(b[hi], c[hi], dmin):
            if dm > MATCH:
                fps.append(dict(model=model, stem=stem, x=cc[0], y=cc[1], conf=bb[4], dist_to_gt=dm))
split_of = split.split
pd_ = pd.DataFrame(rows)
pdf = pd_.merge(obj, on=['stem', 'obj'])
pdf['miss25'] = pdf.conf_at < T
pdf['iou_ok'] = pdf.iou >= 0.5
pdf['split'] = pdf.stem.map(split_of)
pdf['date'] = pdf.stem.map(img.date); pdf['hour'] = pd.to_datetime(pdf.stem.map(img.ts)).dt.hour
pdf['session'] = pdf.stem.map(img.session); pdf['kind'] = np.where(pdf.n_obj == 1, 'k1', 'k3')
pdf['box_rel'] = np.sqrt(pdf.gw * pdf.gh) / pdf.groupby('machine').apply(lambda x: np.sqrt(x.gw * x.gh).median()).reindex(pdf.machine).values
vc = pd.read_csv(f'{K}/eval/marker_v2/variant_check.csv'); vc['obj'] = vc.groupby('stem').cumcount()
pdf = pdf.merge(vc[['stem', 'obj', 'dist_orig']], on=['stem', 'obj'], how='left')
pdf.to_csv(f'{K}/eval/failure/per_defect{TAG}.csv', index=False)
fp = pd.DataFrame(fps)
if len(fp):
    fp['machine'] = fp.stem.map(img.machine)
    pm_cache = {}
    def where(r):
        if r.stem not in pm_cache:
            im = cv2.imread(f'{K}/data/clean/images/{r.stem}.png', 0); pm_cache[r.stem] = (product_mask(im), im)
        pm, im = pm_cache[r.stem]; H, W = pm.shape
        x, y = int(min(max(r.x, 0), W - 1)), int(min(max(r.y, 0), H - 1))
        edt = cv2.distanceTransform(pm.astype(np.uint8), cv2.DIST_L2, 5)
        return pd.Series(dict(in_product=bool(pm[y, x]), dist_prod_edge=float(edt[y, x])))
    fp = pd.concat([fp, fp.apply(where, axis=1)], axis=1)
fp.to_csv(f'{K}/eval/failure/per_fp{TAG}.csv', index=False)
print('결함×모델', len(pdf), '| 결함', pdf.groupby(['stem', 'obj']).ngroups, '| 오검출(conf≥0.25)', len(fp))
