"""미라벨 데이터 기반 참고 평가 (정답 없음 → 정확도가 아닌 일관성 지표)
대상 1) PL평가 날짜 534장: 의사 라벨 학습에 쓰지 않은 날짜 (정상 이미지)
      - 구조 일치율: conf ≥ T 검출 수가 1 또는 3 인 이미지 비율 (테스트피스 구조)
      - 검출 0개 비율: 불량 판정 영상인데 아무것도 못 찾은 비율 (미탐지 의심)
      - 이미지별 최대 conf 중앙값, 날짜별 구조 일치율의 최솟값 (기간 안정성)
대상 2) 이상 107장 (제품 잘림·없음 등): 검출이 나온 이미지 비율 = 헛검출 의심
임계값: 고정 0.25 와 모델별 T99 (val 정밀도 ≥ 0.99 가 되는 최소 conf, 공식 TXT 기준)
게이팅: gate=제품 영역 밖 검출 제외 후 지표 / raw=제외 없이
사용: python3 eval/pseudo/eval_unlabeled.py [run ...]
출력: eval/pseudo/out/unlabeled_eval.csv, unlabeled_dets_<run>.csv
"""
import os, sys
import numpy as np
import pandas as pd
import cv2

K = '/data/knhyun/KAMP'
sys.path.insert(0, f'{K}/eval')
import argparse
ap = argparse.ArgumentParser()
ap.add_argument('runs', nargs='*', default=['B_clean416', 'E_aug416', 'G_pl25', 'G_pl37', 'G_pl46'])
ap.add_argument('--roles', default='data/splits/unlabeled_roles.csv')   # 미라벨 역할 파일 (prep/build_unlabeled.py)
ap.add_argument('--metrics', default='eval/out')                            # 공식 val 검출 결과 폴더 (T99 계산용)
ap.add_argument('--out', default='eval/pseudo/out')
A = ap.parse_args(); RUNS = A.runs
sys.argv = ['x']
exec(open(f'{K}/eda/extract.py').read().split('# ─────────────────────────── 1.')[0])   # product_mask
import evaluate as E

OUT = f'{K}/{A.out}'; os.makedirs(OUT, exist_ok=True)
meta = pd.read_csv(f'{K}/{A.roles}')
target = meta[meta.role.isin(['PL평가', '이상'])]
cfg = {r: (w, z) for r, w, z in E.MODELS}
cfg_of = lambda r: cfg.get(r, (f'{K}/runs/{r}/weights/best.pt', 416))


def t99(run):
    d = pd.read_csv(f'{K}/{A.metrics}/dets_{run.replace("/", "__")}_clean_val.csv').sort_values('conf', ascending=False)
    tp = d.tp_ctr.values.astype(float); prec = np.cumsum(tp) / np.arange(1, len(tp) + 1)
    ok = np.where(prec >= 0.99)[0]
    return float(d.conf.values[ok[-1]]) if len(ok) else 1.0


rows, PM = [], {}
for r in RUNS:
    w, z = cfg_of(r)
    m = E.load_model(w)
    det_rows = []
    for _, u in target.iterrows():
        img = cv2.imread(f'{K}/data/unlabeled/images/{u["name"]}.png', cv2.IMREAD_GRAYSCALE)
        _, det = E.predict_img(m, img, z)
        pm = PM.setdefault(u['name'], product_mask(img))
        dl = det.tolist() if len(det) else []
        confs = [d[4] for d in dl]
        H, W = pm.shape
        inside = [bool(pm[min(int((d[1] + d[3]) / 2), H - 1), min(int((d[0] + d[2]) / 2), W - 1)]) for d in dl]
        det_rows.append(dict(name=u['name'], role=u.role, machine=u.machine, date=u.date, confs=confs,
                             confs_gate=[c for c, i in zip(confs, inside) if i]))
    d = pd.DataFrame(det_rows)
    fmt = lambda c: ';'.join(f'{x:.4f}' for x in c)
    d.assign(confs=d.confs.map(fmt), confs_gate=d.confs_gate.map(fmt)).to_csv(f'{OUT}/unlabeled_dets_{r.replace("/", "__")}.csv', index=False)
    for (tname, T), gate in [(t, g) for t in [('0.25', 0.25), ('T99', t99(r))] for g in ['raw', 'gate']]:
        col = d.confs if gate == 'raw' else d.confs_gate
        n = col.map(lambda c: sum(x >= T for x in c))
        mx = col.map(lambda c: max(c) if c else 0.0)
        ev, ab = d.role == 'PL평가', d.role == '이상'
        by_date = (n[ev].isin([1, 3])).groupby(d.date[ev]).mean()
        row = dict(run=r, thr=tname, T=round(T, 3), gate=gate,
                   structure_ok=(n[ev].isin([1, 3])).mean(), zero_det=(n[ev] == 0).mean(),
                   extra_det=(n[ev].isin([2]) | (n[ev] >= 4)).mean(), max_conf_median=mx[ev].median(),
                   worst_date_structure=by_date.min(), abnormal_any_det=(n[ab] > 0).mean())
        for mc in ['1호기', '2호기', '3호기']:
            row[f'structure_ok_{mc}'] = (n[ev & (d.machine == mc)].isin([1, 3])).mean()
        rows.append(row)
    print(f'{r} 완료')
res = pd.DataFrame(rows)
res.to_csv(f'{OUT}/unlabeled_eval.csv', index=False)
pd.set_option('display.width', 220)
print(res.round(3).to_string(index=False))
