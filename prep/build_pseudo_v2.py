"""미라벨 요인 실험용 의사 라벨 (분할 v2, 베이스라인 교사)
설계: 다른 요인은 베이스라인(B: 인페인팅, 합성 증강 없음, YOLOv3-SPP 416)으로 고정하고 '미라벨 사용 방식'만 바꾼다.
  교사 = 베이스라인 B (seed 별), 학생 = B 설정 + 의사 라벨 이미지 (합성 증강 없음)
미라벨 이미지·역할은 prep/build_unlabeled.py 가 생성 (data/unlabeled/images, data/splits/unlabeled_roles.csv).
의사 라벨 대상 = 역할 'PL학습'
의사 라벨: 교사 val(공식 TXT) 정밀도 목표(0.99, 1.00)를 만족하는 최소 conf 를 임계값 T 로.
  필터: 제품 영역 밖 검출 제외 / 검출 수 1 또는 3 / 모호 구간 [0.10, T) 검출 있으면 이미지 제외
사용: python3 prep/build_pseudo_v2.py --teacher v2/B_s0 --tag s0
출력: data/pl_v2/<tag>_p<목표>/{images→symlink, labels}, data/splits/pl_v2_<tag>_p<목표>_train.txt, data/pl_v2_<tag>_p<목표>.data
"""
import os, sys, shutil, argparse
import numpy as np
import pandas as pd
import cv2

K = '/data/knhyun/KAMP'
ap = argparse.ArgumentParser()
ap.add_argument('--teacher', required=True); ap.add_argument('--tag', required=True)
ap.add_argument('--targets', nargs='*', type=float, default=[0.99, 1.00])
args = ap.parse_args()
sys.path.insert(0, f'{K}/eval'); sys.argv = ['x']
exec(open(f'{K}/eda/extract.py').read().split('# ─────────────────────────── 1.')[0])   # product_mask
import evaluate as E

T_AMB = 0.10
DATA, OUT = f'{K}/data', f'{K}/eval/pseudo_v2'
UL_IMG = f'{DATA}/unlabeled/images'
os.makedirs(OUT, exist_ok=True)

# ── 미라벨 역할: prep/build_unlabeled.py 가 생성 (이름·품질·역할)
meta = pd.read_csv(f'{DATA}/splits/unlabeled_roles.csv')

# ── 교사 추론 (교사별 캐시)
tname = args.teacher.replace('/', '__')
det_f = f'{OUT}/teacher_dets_{tname}.csv'
if not os.path.exists(det_f):
    model = E.load_model(f'{K}/runs/{args.teacher}/weights/best.pt')
    rows = []
    for _, r in meta[meta.role.isin(['PL학습', 'PL평가'])].iterrows():
        img = cv2.imread(f'{UL_IMG}/{r["name"]}.png', cv2.IMREAD_GRAYSCALE)
        _, det = E.predict_img(model, img, 416)
        pm = product_mask(img); H, W = pm.shape
        for d in det.tolist():
            cx, cy = min(int((d[0] + d[2]) / 2), W - 1), min(int((d[1] + d[3]) / 2), H - 1)
            rows.append(dict(name=r['name'], x1=d[0], y1=d[1], x2=d[2], y2=d[3], conf=d[4], in_product=bool(pm[cy, cx])))
    pd.DataFrame(rows, columns=['name', 'x1', 'y1', 'x2', 'y2', 'conf', 'in_product']).to_csv(det_f, index=False)
dets = pd.read_csv(det_f)

# ── 교사 val 정밀도로 임계값 결정 (공식 TXT)
vd = pd.read_csv(f'{K}/eval/out_v2/dets_{tname}_clean_val.csv').sort_values('conf', ascending=False)
tp = vd.tp_ctr.values.astype(float); prec = np.cumsum(tp) / np.arange(1, len(tp) + 1)
thresholds = {}
for tgt in args.targets:
    ok = np.where(prec >= tgt - 1e-9)[0]
    # 정밀도 목표를 만족하는 가장 낮은 conf (그 지점까지의 누적 정밀도 ≥ 목표)
    thresholds[f'p{int(round(tgt * 1000))}'] = float(vd.conf.values[ok[-1]]) if len(ok) else 1.0

gt_train = [f'{DATA}/clean/images/{s}.png' for s in pd.read_csv(f'{DATA}/splits/split.csv').query('split=="train"').stem]
pool = meta[meta.role == 'PL학습']
summary = []
for key, T in thresholds.items():
    name = f'{args.tag}_{key}'
    root = f'{DATA}/pl_v2/{name}'
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(f'{root}/images'); os.makedirs(f'{root}/labels')
    kept, n_obj, drop = [], 0, {'count': 0, 'ambiguous': 0}
    for _, r in pool.iterrows():
        d = dets[(dets.name == r['name']) & dets.in_product]
        hi, amb = d[d.conf >= T], d[(d.conf >= T_AMB) & (d.conf < T)]
        if len(hi) not in (1, 3): drop['count'] += 1; continue
        if len(amb): drop['ambiguous'] += 1; continue
        lines = [f'0 {(a.x1 + a.x2) / 2 / r.w:.6f} {(a.y1 + a.y2) / 2 / r.h:.6f} {(a.x2 - a.x1) / r.w:.6f} {(a.y2 - a.y1) / r.h:.6f}' for a in hi.itertuples()]
        os.symlink(f'{UL_IMG}/{r["name"]}.png', f'{root}/images/{r["name"]}.png')
        open(f'{root}/labels/{r["name"]}.txt', 'w').write('\n'.join(lines) + '\n')
        kept.append(f'{root}/images/{r["name"]}.png'); n_obj += len(lines)
    lst = f'{DATA}/splits/pl_v2_{name}_train.txt'
    open(lst, 'w').write('\n'.join(gt_train + kept) + '\n')
    open(f'{DATA}/pl_v2_{name}.data', 'w').write(f'classes=1\ntrain={lst}\nvalid={DATA}/splits/clean_val.txt\nnames={DATA}/classes.names\n')
    summary.append(dict(teacher=args.teacher, name=name, T=round(T, 4), pool=len(pool), kept=len(kept), pl_objects=n_obj,
                        drop_count=drop['count'], drop_ambiguous=drop['ambiguous'], train_total=len(gt_train) + len(kept)))
s = pd.DataFrame(summary)
f = f'{OUT}/pl_summary.csv'
if os.path.exists(f):
    prev = pd.read_csv(f); s = pd.concat([prev[~prev.name.isin(s.name)], s])
s.to_csv(f, index=False)
print(s[s.teacher == args.teacher].to_string(index=False))
