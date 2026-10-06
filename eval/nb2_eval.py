"""새 베이스라인(YOLOv8s, D-FINE-S) 기본 평가: val·test, 공식 TXT 기준 (YOLOv3 평가와 같은 매칭·지표, eval/evaluate.py 재사용)
사용: python eval/nb2_eval.py [yolov8s dfine_s]   → eval/out_nb2/metrics.csv, per_machine.csv, dets_*.csv, 콘솔 요약 (seed 평균 ± 표준편차)
입력 해상도 640, 입력 = data/clean (Telea 인페인팅, handoff 패키지 이미지와 동일)
"""
import os, sys
import pandas as pd
import torch
K = '/data/knhyun/KAMP'; sys.path.insert(0, f'{K}/eval'); sys.argv = sys.argv[:1] + [a for a in sys.argv[1:]]
models = sys.argv[1:] or ['yolov8s', 'dfine_s']; sys.argv = ['x']
import evaluate as E
import nb2_models as NB
NB.patch(E)
E.OUT = f'{K}/eval/out_nb2'; os.makedirs(E.OUT, exist_ok=True)
rows = []
for mname in models:
    for run, spec in NB.specs(mname).items():
        if not os.path.exists(spec.split('|')[-1].split(':', 1)[-1]):
            print('skip (없음):', spec); continue
        m = E.load_model(spec)
        for sp in ['val', 'test']:
            r = E.evaluate(run, spec, 640, 'clean', sp, m); rows += r
            print(f'{run:20s} {sp:4s} AP50={r[0]["AP50"]:.4f} P@.25={r[0]["P_at25"]:.3f} R@.25={r[0]["R_at25"]:.3f} ctrAP={r[0]["ctr_AP50"]:.4f}', flush=True)
        del m; torch.cuda.empty_cache()
D = pd.DataFrame(rows)
if os.path.exists(f'{E.OUT}/per_machine.csv'):   # 다른 모델 결과는 유지하고 이번에 평가한 run 만 교체
    prev = pd.read_csv(f'{E.OUT}/per_machine.csv'); D = pd.concat([prev[~prev.run.isin(D.run)], D], ignore_index=True)
D[D.scope == 'all'].drop(columns='scope').to_csv(f'{E.OUT}/metrics.csv', index=False); D.to_csv(f'{E.OUT}/per_machine.csv', index=False)
# YOLOv3 베이스라인(v2/B_s*)과 나란히
old = pd.read_csv(f'{K}/eval/out_v2/per_machine.csv'); old = old[old.run.str.startswith('v2/B_s') & (old.input == 'clean')]
A = pd.concat([D.assign(model=D.run.str.replace(r'_s\d$', '', regex=True)), old.assign(model='v2/yolov3spp_416')])
cols = ['AP50', 'P_at25', 'R_at25', 'ctr_AP50']
S = A.groupby(['split', 'scope', 'model'])[cols].agg(['mean', 'std']).round(3)
pd.set_option('display.width', 220); print(S.to_string()); S.to_csv(f'{E.OUT}/summary.csv')
