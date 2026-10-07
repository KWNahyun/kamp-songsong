"""판정 기준(요구사항 ③)용 검출 수집: 후보 모델 seed 3개 × 영상 묶음, 신뢰도 ≥ 0.001 전부 + 제품 영역 안/밖 표시
영상 묶음:
  real    : GT val·test 150장 (data/clean, 마커 Telea 인페인팅) — 실제 결함 위치(공식 TXT 중심)와 중심거리 매칭
  erased  : 같은 150장에서 실제 결함까지 지운 영상 (data/synth_val/erased) — '결함 없는 제품' 대용 (지운 흔적 있음, §3.8 V4)
  unlab   : GT 없는 2,020장 (data/unlabeled/images) — 실제 운영 흐름의 재검사율 추정 (라벨 없음)
제품 영역: eda/extract.py product_mask 를 4px 넓힌 것 (경계에 걸친 검출을 살리기 위해). 결함은 모두 제품 가장자리에서 36px 이상 안쪽(§3.6)
사용: python eval/decision/collect_dets.py <변형 ...>   (기본 yolov8s yolov8s_tpB yolov8s_tpC)
출력: eval/decision/dets_<변형>.csv (set, run, stem, cx, cy, w, h, conf, in_prod, gt_match, gt_id), eval/decision/gts.csv (val·test 결함 중심)
"""
import os, sys
import numpy as np
import pandas as pd
import cv2
from scipy import ndimage as ndi

K = '/data/knhyun/KAMP'; OUT = f'{K}/eval/decision'; os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, f'{K}/eval'); argv = sys.argv[1:] or ['yolov8s', 'yolov8s_tpB', 'yolov8s_tpC']; sys.argv = ['x']
import evaluate as E
import nb2_models as NB
NB.patch(E)
CENTER_T = 8.0


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
unl = sorted(os.path.splitext(f)[0] for f in os.listdir(f'{K}/data/unlabeled/images'))
HO = f'{K}/handoff/kamp_xray_v2'
gts = []
for s in vt:
    im = cv2.imread(f'{K}/data/clean/images/{s}.png', 0); H, W = im.shape
    for i, l in enumerate(open(f'{HO}/labels/{split.split[s]}/{s}.txt')):
        c, x, y, w, h = map(float, l.split())
        gts.append(dict(stem=s, split=split.split[s], machine=split.machine[s], gt_id=i, cx=x * W, cy=y * H))
G = pd.DataFrame(gts); G.to_csv(f'{OUT}/gts.csv', index=False)
Gs = {s: g[['cx', 'cy']].values for s, g in G.groupby('stem')}
sets = [('real', vt, lambda s: f'{K}/data/clean/images/{s}.png', lambda s: f'{K}/data/clean/images/{s}.png'),
        ('erased', vt, lambda s: f'{K}/data/synth_val/erased/images/{s}.png', lambda s: f'{K}/data/clean/images/{s}.png'),
        ('unlab', unl, lambda s: f'{K}/data/unlabeled/images/{s}.png', lambda s: f'{K}/data/unlabeled/images/{s}.png')]
masks = {}
for v in argv:
    models = {r: E.load_model(sp) for r, sp in NB.specs(v).items()}
    rows = []
    for name, stems, path, mpath in sets:
        for s in stems:
            key = mpath(s)
            if key not in masks:
                pm = product_mask(cv2.imread(key, 0)); masks[key] = cv2.dilate(pm.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
            pm = masks[key]; H, W = pm.shape
            im = cv2.imread(path(s), 0)
            for run, m in models.items():
                _, det = E.predict_img(m, im, 640); d = det.numpy()
                if not len(d):
                    rows.append(dict(set=name, run=run, stem=s, conf=np.nan)); continue   # 검출 0개인 영상도 남김
                cx, cy = (d[:, 0] + d[:, 2]) / 2, (d[:, 1] + d[:, 3]) / 2
                inp = pm[np.clip(cy.astype(int), 0, H - 1), np.clip(cx.astype(int), 0, W - 1)]
                gm, gid = np.zeros(len(d), bool), np.full(len(d), -1)
                if name == 'real' and s in Gs:      # 중심거리 greedy 매칭 (신뢰도 높은 순, evaluate.match 와 같은 규칙)
                    g = Gs[s]; used = np.zeros(len(g), bool)
                    for i in np.argsort(-d[:, 4]):
                        dist = np.where(used, np.inf, np.hypot(g[:, 0] - cx[i], g[:, 1] - cy[i])); j = int(np.argmin(dist))
                        if dist[j] <= CENTER_T:
                            gm[i], gid[i], used[j] = True, j, True
                for i in range(len(d)):
                    rows.append(dict(set=name, run=run, stem=s, cx=cx[i], cy=cy[i], w=d[i, 2] - d[i, 0], h=d[i, 3] - d[i, 1],
                                     conf=d[i, 4], in_prod=bool(inp[i]), gt_match=bool(gm[i]), gt_id=int(gid[i])))
        print(v, name, 'done', flush=True)
    pd.DataFrame(rows).to_csv(f'{OUT}/dets_{v}.csv', index=False)
# 확인: 모든 GT 결함 중심이 제품 영역 안인가
G['in_prod'] = [masks[f'{K}/data/clean/images/{r.stem}.png'][int(r.cy), int(r.cx)] for r in G.itertuples()]
print('GT 결함 중 제품 영역 밖:', int((~G.in_prod).sum()), '/', len(G))
