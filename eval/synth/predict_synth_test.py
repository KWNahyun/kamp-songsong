"""합성 test 패키지(handoff/kamp_synth_test_v1)에 학습된 모델을 돌려 예측 CSV 저장 → tools/eval_synth.py 로 평가
패키지가 보고서 §4.5·§4.6과 같은 영상인지 확인하는 용도 (모델 담당자에게는 같은 형식의 자기 예측을 받음)
사용: python eval/synth/predict_synth_test.py yolov8s dfine_s   → eval/synth/synth_test/preds_<모델>_s<seed>_<split>.csv
"""
import os, sys
import cv2
import pandas as pd

K = '/data/knhyun/KAMP'
sys.path.insert(0, f'{K}/eval')
VARS = sys.argv[1:] or ['yolov8s', 'dfine_s']; sys.argv = ['x']
import evaluate as E
import nb2_models as NB
NB.patch(E)
PKG = f'{K}/handoff/kamp_synth_test_v1'; OUT = f'{K}/eval/synth/synth_test'
os.makedirs(OUT, exist_ok=True)
I = pd.read_csv(f'{PKG}/images.csv')
for v in VARS:
    for run, sp in NB.specs(v).items():
        m = E.load_model(sp)
        for split in ['val', 'test']:
            rows = []
            for iid in I[I.split == split].image_id:
                _, det = E.predict_img(m, cv2.imread(f'{PKG}/images/{split}/{iid}.png', 0), 640)
                rows += [dict(image_id=iid, x1=float(d[0]), y1=float(d[1]), x2=float(d[2]), y2=float(d[3]), conf=float(d[4])) for d in det.numpy()]
            f = f'{OUT}/preds_{run.split("/")[-1]}_{split}.csv'
            pd.DataFrame(rows).to_csv(f, index=False); print(f, len(rows), flush=True)
