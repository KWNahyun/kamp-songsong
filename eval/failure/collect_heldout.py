"""실패 조건 분석용: GT 500장 전부에 대해 '학습에 안 쓴 모델'의 검출을 모은다
- val·test 150장: 베이스라인 v2/B_s0~2 (seed 3개)
- train 350장: 교차 검증 교사 v2/CV_f<k> (조각 k 는 그 교사가 학습하지 않은 사진, seed 0 하나)
출력: eval/failure/dets_heldout.csv (model, stem, split, x1..conf)
환경변수 NB2_MODEL=yolov8s|dfine_s: 새 베이스라인 seed 3개 (640), val·test 150장만 (교차 검증 교사 없음) → dets_heldout_<모델>.csv
"""
import os, sys
import pandas as pd
import cv2
K = '/data/knhyun/KAMP'
sys.path.insert(0, f'{K}/eval'); sys.argv = ['x']
import evaluate as E

split = pd.read_csv(f'{K}/data/splits/split.csv'); cv = pd.read_csv(f'{K}/data/splits/cv3.csv')
jobs = [(f'v2/B_s{s}', split[split.split.isin(['val', 'test'])].stem.tolist()) for s in range(3)]
jobs += [(f'v2/CV_f{k}', cv[cv.fold == k].stem.tolist()) for k in range(3)]
NB2 = os.environ.get('NB2_MODEL'); SIZE, TAG = 416, ''
spec_of = lambda run: f'{K}/runs/{run}/weights/best.pt'
if NB2:
    import nb2_models as NB
    NB.patch(E); sp_ = NB.specs(NB2); SIZE, TAG = 640, f'_{NB2}'
    jobs = [(r, split[split.split.isin(['val', 'test'])].stem.tolist()) for r in sp_]; spec_of = sp_.get
rows = []
for run, stems in jobs:
    m = E.load_model(spec_of(run))
    for s in stems:
        _, det = E.predict_img(m, cv2.imread(f'{K}/data/clean/images/{s}.png', 0), SIZE)
        for d in det.numpy():
            rows.append(dict(model=run, stem=s, x1=d[0], y1=d[1], x2=d[2], y2=d[3], conf=d[4]))
    print(run, len(stems))
pd.DataFrame(rows).to_csv(f'{K}/eval/failure/dets_heldout{TAG}.csv', index=False)
