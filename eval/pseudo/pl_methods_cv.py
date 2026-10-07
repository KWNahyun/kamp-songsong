"""의사 라벨 방법 비교 (마커 / 교사 / 혼합) — train 3-fold 교차 검증, 공식 TXT 로 평가
조각 k: 교사 = runs/v2/CV_f<k> (나머지 두 조각으로 학습), 평가 = 조각 k 의 GT 이미지
임계값 T = 교사의 공식 val 정밀도 ≥ 0.99 가 되는 최소 conf (중심거리 매칭)
방법:
  MK : 원본 컬러 마커 연결요소의 중심 → 박스 (크기 = 교사 학습 조각의 장비별 GT 박스 중앙값). 제품 밖 마커 제외
  T0 : 교사 conf ≥ T
  T1 : T0 + 제품 영역 밖 검출 제외
  T2 : T1 + 이미지 검출 수가 1 또는 3 인 이미지만
  T3 : T2 + 모호 구간 [0.10, T) 검출이 있는 이미지 제외   (현재 쓰는 방식)
  T4 : T3 + 세션 일관성 (같은 세션 이미지들의 T1 검출 수 최빈값과 같은 이미지만. 세션에 1장뿐이면 T3 그대로)
  H1 : 교사 검출(conf ≥ 0.05, 제품 안) 중 마커 중심이 8px 안에 있는 것만
  H2 : 마커 위치마다 교사 박스(conf ≥ 0.01)가 8px 안에 있으면 그 박스, 없으면 MK 박스
출력: eval/pseudo_v2/pl_methods_cv.csv (방법 × 조각 지표), pl_methods_cv_boxes.csv (박스 단위 기록)
"""
import os, sys
import numpy as np
import pandas as pd
import cv2
import torch
from scipy import ndimage as ndi

K = '/data/knhyun/KAMP'
sys.path.insert(0, f'{K}/eval'); sys.argv = ['x']
exec(open(f'{K}/eda/extract.py').read().split('# ─────────────────────────── 1.')[0])   # load_rgb, product_mask
import evaluate as E

T_AMB, T_H1, T_H2, MATCH = 0.10, 0.05, 0.01, 8.0
OUT = f'{K}/eval/pseudo_v2'
cv = pd.read_csv(f'{K}/data/splits/cv3.csv')
split = pd.read_csv(f'{K}/data/splits/split.csv')
raw = pd.read_csv(f'{K}/eda/out/raw_inventory.csv').groupby('stem').path.first()
val = split[split.split == 'val'].stem.tolist()


def gt_boxes(stem, W, H):
    return E.load_gt('clean', stem, W, H).numpy()


def centers(b):
    return (b[:, :2] + b[:, 2:4]) / 2


def match_boxes(pred, gt):
    """pred: (n,5) x1y1x2y2conf. conf 내림차순 greedy. 반환: IoU 기준 TP, 중심 기준 TP, 중심 기준 매칭된 GT"""
    n, m = len(pred), len(gt)
    tp_iou, tp_ctr, g_ctr = np.zeros(n, bool), np.zeros(n, bool), np.zeros(m, bool)
    if n and m:
        det = torch.tensor(pred, dtype=torch.float32); g = torch.tensor(gt, dtype=torch.float32)
        ti, tc, _, gc, _ = E.match(det, g)
        tp_iou, tp_ctr, g_ctr = ti, tc, gc >= 0
    return tp_iou, tp_ctr, g_ctr


def t99(model):
    rows = []
    for s in val:
        img = cv2.imread(f'{K}/data/clean/images/{s}.png', 0); H, W = img.shape
        _, det = E.predict_img(model, img, 416)
        d = det.numpy(); g = gt_boxes(s, W, H)
        _, tc, _ = match_boxes(d, g)
        rows += list(zip(d[:, 4], tc))
    rows.sort(key=lambda x: -x[0])
    tp = np.array([r[1] for r in rows], float); prec = np.cumsum(tp) / np.arange(1, len(tp) + 1)
    ok = np.where(prec >= 0.99)[0]
    return float(rows[ok[-1]][0]) if len(ok) else 1.0


records, boxes_log = [], []
for f in sorted(cv.fold.unique()):
    model = E.load_model(f'{K}/runs/v2/CV_f{f}/weights/best.pt')
    T = t99(model)
    tr_stems = cv[cv.fold != f].stem
    # 장비별 GT 박스 크기 중앙값 (교사 학습 조각의 공식 라벨) → 마커 박스 크기
    sz = {}
    for m, g in cv[cv.fold != f].groupby('machine'):
        wh = []
        for s in g.stem:
            img = cv2.imread(f'{K}/data/clean/images/{s}.png', 0); H, W = img.shape
            b = gt_boxes(s, W, H); wh += list(zip(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]))
        sz[m] = np.median(np.array(wh), 0)
    hold = cv[cv.fold == f]
    per_img = {}
    for r in hold.itertuples():
        img = cv2.imread(f'{K}/data/clean/images/{r.stem}.png', 0); H, W = img.shape
        pm = product_mask(img)
        inside = lambda c: bool(pm[min(int(c[1]), H - 1), min(int(c[0]), W - 1)])
        _, det = E.predict_img(model, img, 416); det = det.numpy()
        gt = gt_boxes(r.stem, W, H)
        # 마커 연결요소
        _, col, _ = load_rgb(raw[r.stem])
        lab, n = ndi.label(col, structure=np.ones((3, 3)))
        mk = []
        for sl in ndi.find_objects(lab):
            cx, cy = (sl[1].start + sl[1].stop) / 2, (sl[0].start + sl[0].stop) / 2
            if inside((cx, cy)):
                w, h = sz[r.machine]
                mk.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, 1.0])
        mk = np.array(mk).reshape(-1, 5)
        per_img[r.stem] = dict(det=det, gt=gt, mk=mk, session=r.session, machine=r.machine,
                               ins=np.array([inside(c) for c in centers(det)]) if len(det) else np.zeros(0, bool))
    # 세션 최빈 검출 수 (T1 기준)
    n_t1 = {s: int(((v['det'][:, 4] >= T) & v['ins']).sum()) if len(v['det']) else 0 for s, v in per_img.items()}
    ses_mode = pd.Series({s: per_img[s]['session'] for s in per_img}).to_frame('session').assign(n=pd.Series(n_t1)) \
        .groupby('session').n.agg(lambda x: (x.mode().iloc[0], len(x)))

    def method(name, s, v):
        d, ins = v['det'], v['ins']
        hi = (d[:, 4] >= T) if len(d) else np.zeros(0, bool)
        if name == 'MK':
            return v['mk'], True
        if name == 'T0':
            return d[hi], True
        if name == 'T1':
            return d[hi & ins], True
        keep = n_t1[s] in (1, 3)
        if name == 'T2':
            return d[hi & ins], keep
        amb = ((d[:, 4] >= T_AMB) & (d[:, 4] < T) & ins).any() if len(d) else False
        keep = keep and not amb
        if name == 'T3':
            return d[hi & ins], keep
        if name == 'T4':
            mode, cnt = ses_mode[v['session']]
            return d[hi & ins], keep and (cnt < 2 or n_t1[s] == mode)
        if name == 'H1':
            sel = (d[:, 4] >= T_H1) & ins if len(d) else np.zeros(0, bool)
            c = d[sel]
            if len(c) and len(v['mk']):
                dist = np.linalg.norm(centers(c)[:, None] - centers(v['mk'])[None], axis=2)
                c = c[dist.min(1) <= MATCH]
            else:
                c = c[:0]
            return c, True
        if name == 'H2':
            out = []
            cand = d[d[:, 4] >= T_H2] if len(d) else d
            for b in v['mk']:
                if len(cand):
                    dist = np.linalg.norm(centers(cand) - centers(b[None])[0], axis=1)
                    j = np.argmin(dist)
                    if dist[j] <= MATCH:
                        out.append(np.r_[cand[j, :4], 1.0]); continue
                out.append(b)
            return np.array(out).reshape(-1, 5), True

    for name in ['MK', 'T0', 'T1', 'T2', 'T3', 'T4', 'H1', 'H2']:
        n_img = kept = n_pl = tp_i = tp_c = n_gt_all = n_gt_kept = hit_kept = exact = 0
        for s, v in per_img.items():
            n_img += 1; n_gt_all += len(v['gt'])
            b, keep = method(name, s, v)
            if not keep:
                continue
            kept += 1
            ti, tc, gc = match_boxes(b, v['gt'])
            n_pl += len(b); tp_i += int(ti.sum()); tp_c += int(tc.sum())
            n_gt_kept += len(v['gt']); hit_kept += int(gc.sum())
            exact += int(tc.all() and gc.all())
            for bb, a, c in zip(b, ti, tc):
                boxes_log.append(dict(fold=f, method=name, stem=s, machine=v['machine'], w=bb[2] - bb[0], h=bb[3] - bb[1],
                                      tp_iou=bool(a), tp_ctr=bool(c)))
        records.append(dict(fold=f, method=name, T=round(T, 4), images=n_img, kept=kept, gt_all=n_gt_all, pl_boxes=n_pl,
                            prec_iou=tp_i / max(n_pl, 1), prec_ctr=tp_c / max(n_pl, 1),
                            recall_in_kept=hit_kept / max(n_gt_kept, 1), missed_in_kept=n_gt_kept - hit_kept,
                            exact_img=exact / max(kept, 1), coverage=hit_kept / max(n_gt_all, 1), tp_iou=tp_i, tp_ctr=tp_c,
                            gt_kept=n_gt_kept, hit_kept=hit_kept, n_exact=exact))
    del model; torch.cuda.empty_cache()
    print(f'fold {f} 완료 (T={T:.3f})')

df = pd.DataFrame(records); df.to_csv(f'{OUT}/pl_methods_cv.csv', index=False)
pd.DataFrame(boxes_log).to_csv(f'{OUT}/pl_methods_cv_boxes.csv', index=False)
# 세 조각 합산
g = df.groupby('method').sum(numeric_only=True)
tot = pd.DataFrame({
    '채택 이미지': (g.kept / g.images).map('{:.1%}'.format),
    '박스 정밀도(IoU≥0.5)': (g.tp_iou / g.pl_boxes).round(3), '박스 정밀도(중심)': (g.tp_ctr / g.pl_boxes).round(3),
    '채택 이미지 재현율': (g.hit_kept / g.gt_kept).round(3), '채택 이미지 놓친 결함': g.gt_kept - g.hit_kept,
    '완전 일치 이미지': (g.n_exact / g.kept).round(3), '전체 커버리지': (g.hit_kept / g.gt_all).round(3),
}).reindex(['MK', 'T0', 'T1', 'T2', 'T3', 'T4', 'H1', 'H2'])
pd.set_option('display.width', 220)
print(tot.to_string())
tot.to_csv(f'{OUT}/pl_methods_cv_summary.csv')
