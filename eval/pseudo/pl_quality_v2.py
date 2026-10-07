"""의사 라벨 품질 추정 (정답이 있는 val 에 미라벨과 같은 절차를 적용해 공식 TXT 와 비교)
절차 = build_pseudo_v2.py 와 동일: 교사 검출 → 제품 영역 밖 제외 → conf ≥ T → 검출 수 1 또는 3 → 모호 구간 [0.10, T) 검출 있으면 제외
지표:
  keep_rate          : 필터를 통과해 의사 라벨 이미지로 쓰이는 비율
  pl_precision       : 의사 라벨 박스 중 실제 결함(중심 ≤ 8px)인 비율
  pl_recall_in_kept  : 통과한 이미지의 실제 결함 중 라벨이 붙은 비율 (나머지는 '배경'으로 잘못 학습됨)
  img_exact          : 통과한 이미지 중 라벨이 정답과 완전히 일치(누락·오검출 없음)하는 비율
주의: 임계값을 val 에서 정했으므로 val 기준 품질은 다소 낙관적
사용: python3 eval/pseudo/pl_quality_v2.py v2/B_s0 v2/B_s1 v2/B_s2
출력: eval/pseudo_v2/pl_quality.csv
"""
import os, sys
import numpy as np
import pandas as pd
import cv2

K = '/data/knhyun/KAMP'
teachers = sys.argv[1:]
sys.argv = ['x']
exec(open(f'{K}/eda/extract.py').read().split('# ─────────────────────────── 1.')[0])   # product_mask

T_AMB, MATCH, TARGETS = 0.10, 8.0, [0.99, 1.00]
split = pd.read_csv(f'{K}/data/splits/split.csv')
val = split[split.split == 'val'].stem.tolist()
PM = {s: product_mask(cv2.imread(f'{K}/data/clean/images/{s}.png', cv2.IMREAD_GRAYSCALE)) for s in val}

rows = []
for t in teachers:
    tn = t.replace('/', '__')
    dets = pd.read_csv(f'{K}/eval/out_v2/dets_{tn}_clean_val.csv')
    gts = pd.read_csv(f'{K}/eval/out_v2/gts_{tn}_clean_val.csv')
    vd = dets.sort_values('conf', ascending=False)
    tp = vd.tp_ctr.values.astype(float); prec = np.cumsum(tp) / np.arange(1, len(tp) + 1)
    for tgt in TARGETS:
        ok = np.where(prec >= tgt - 1e-9)[0]
        T = float(vd.conf.values[ok[-1]]) if len(ok) else 1.0
        n_img = kept = n_pl = n_pl_ok = n_gt_kept = n_gt_hit = exact = 0
        drop = {'count': 0, 'ambiguous': 0}
        for s in val:
            n_img += 1
            pm = PM[s]; H, W = pm.shape
            d = dets[dets.stem == s].copy()
            d['cx'] = ((d.x1 + d.x2) / 2).clip(0, W - 1); d['cy'] = ((d.y1 + d.y2) / 2).clip(0, H - 1)
            d = d[[bool(pm[int(y), int(x)]) for x, y in zip(d.cx, d.cy)]]
            hi, amb = d[d.conf >= T], d[(d.conf >= T_AMB) & (d.conf < T)]
            if len(hi) not in (1, 3): drop['count'] += 1; continue
            if len(amb): drop['ambiguous'] += 1; continue
            kept += 1
            g = gts[gts.stem == s]
            dist = np.hypot(hi.cx.values[:, None] - g.cx.values[None], hi.cy.values[:, None] - g.cy.values[None]) if len(g) else np.zeros((len(hi), 0))
            pl_ok = (dist <= MATCH).any(1) if dist.size else np.zeros(len(hi), bool)
            gt_hit = (dist <= MATCH).any(0) if dist.size else np.zeros(len(g), bool)
            n_pl += len(hi); n_pl_ok += int(pl_ok.sum()); n_gt_kept += len(g); n_gt_hit += int(gt_hit.sum())
            exact += int(pl_ok.all() and gt_hit.all())
        rows.append(dict(teacher=t, target=tgt, T=round(T, 4), val_images=n_img, keep_rate=kept / n_img,
                         drop_count=drop['count'], drop_ambiguous=drop['ambiguous'],
                         pl_precision=n_pl_ok / max(n_pl, 1), pl_recall_in_kept=n_gt_hit / max(n_gt_kept, 1),
                         img_exact=exact / max(kept, 1), n_pl=n_pl, gt_missed_in_kept=n_gt_kept - n_gt_hit))
df = pd.DataFrame(rows)
os.makedirs(f'{K}/eval/pseudo_v2', exist_ok=True)
df.to_csv(f'{K}/eval/pseudo_v2/pl_quality.csv', index=False)
pd.set_option('display.width', 200)
print(df.round(3).to_string(index=False))
