"""합성 test 평가: 조건(세트)별 검출률, 오검출, 결함 없는 영상 반응
입력: 예측 CSV (image_id,x1,y1,x2,y2,conf; 원본 픽셀 좌표, NMS 후 conf ≥ 0.001 권장) — 메인 패키지 tools/evaluate.py 와 같은 형식
찾음 규칙 (보고서 §4.5·§4.6과 같음):
  신뢰도 ≥ thr 인 검출의 중심이 물체 중심 8px 안이거나, 평가 영역(이물 영역을 4px 넓힌 사각형, objects.csv 의 ex1..ex2) 안 → 찾음
  물체의 신뢰도 = 위 조건을 만족하는 검출 중 최고 신뢰도 (없으면 0)
오검출: 신뢰도 ≥ thr 인데 어느 물체와도 짝이 안 된 검출 (영상당 평균)
erased 세트(결함 없는 제품 대용): 신뢰도 ≥ thr 검출이 하나라도 있는 영상 비율
사용 예:
  python tools/eval_synth.py --pred my_preds_test.csv --split test
  python tools/eval_synth.py --pred my_preds_test.csv --split test --thr 0.25 0.5 --by machine --out per_object.csv
"""
import argparse, os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument('--pred', required=True)
ap.add_argument('--split', default='test', choices=['val', 'test'])
ap.add_argument('--thr', type=float, nargs='+', default=[0.25])
ap.add_argument('--by', default=None, help='추가로 나눠 볼 열 (예: machine)')
ap.add_argument('--out', default=None, help='물체별 신뢰도 CSV 저장 경로')
a = ap.parse_args()

I = pd.read_csv(f'{ROOT}/images.csv'); O = pd.read_csv(f'{ROOT}/objects.csv')
I = I[I.split == a.split]; O = O[O.split == a.split].copy()
D = pd.read_csv(a.pred); D['image_id'] = D.image_id.astype(str).str.replace(r'\.png$', '', regex=True)
missing = set(I.image_id) - set(D.image_id)
if missing:
    print(f'주의: 예측이 하나도 없는 영상 {len(missing)}장 (검출 0개로 처리). 예: {sorted(missing)[:3]}')
D = D[D.image_id.isin(I.image_id)]
D['cx'] = (D.x1 + D.x2) / 2; D['cy'] = (D.y1 + D.y2) / 2
dg = {k: v for k, v in D.groupby('image_id')}

conf = np.zeros(len(O)); matched = {}
for i, (iid, o) in enumerate(zip(O.image_id.values, O.itertuples())):
    d = dg.get(iid)
    if d is None:
        continue
    fx, fy = np.floor(d.cx.values), np.floor(d.cy.values)
    near = (np.hypot(d.cx.values - o.obj_cx, d.cy.values - o.obj_cy) <= 8) | \
           ((fx >= o.ex1) & (fx < o.ex2) & (fy >= o.ey1) & (fy < o.ey2))
    if near.any():
        conf[i] = d.conf.values[near].max()
        matched.setdefault(iid, np.zeros(len(d), bool))
        matched[iid] |= near
O['conf'] = conf
if a.out:
    O.to_csv(a.out, index=False)

pd.set_option('display.width', 200); pd.set_option('display.max_rows', 200)
order = list(dict.fromkeys(I.set))
for thr in a.thr:
    print(f'\n===== 신뢰도 임계값 {thr} ({a.split}) =====')
    keys = ['set'] + ([a.by] if a.by else [])
    det = O.assign(found=O.conf >= thr).groupby(keys, sort=False).agg(objects=('found', 'size'), detection_rate=('found', 'mean'))
    fp = []
    for iid, s in zip(I.image_id, I.set):
        d = dg.get(iid)
        hi = (d.conf.values >= thr) if d is not None else np.zeros(0, bool)
        m = matched.get(iid, np.zeros(len(hi), bool))
        fp.append(dict(image_id=iid, set=s, n_fp=int((hi & ~m).sum()), any_det=bool(hi.any())))
    F = pd.DataFrame(fp).merge(I[['image_id', 'machine']], on='image_id')
    fpk = ['set'] + ([a.by] if a.by else [])
    fps = F.groupby(fpk, sort=False).agg(images=('n_fp', 'size'), fp_per_image=('n_fp', 'mean'), any_det_rate=('any_det', 'mean'))
    out = fps.join(det, how='left').reset_index()
    out['set'] = pd.Categorical(out.set, order, ordered=True); out = out.sort_values(keys)
    print(out.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    print('  erased: any_det_rate = 결함 없는 영상에서 검출이 나온 비율 (낮을수록 좋음)')
