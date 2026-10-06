"""모델 변형 비교표: 사전학습 조건(① 없음 ② COCO ③ COCO→PIDray)과 옮겨 심기 학습(tpB·tpC, 대조 tpX)
지표 (seed 3개 평균 ± 표준편차):
  실제 test AP@0.5 (전체, 3호기), 실제 val AP@0.5            ← eval/out_nb2/per_machine.csv (nb2_eval.py)
  지운 자리 반응 (신뢰도 ≥ 0.25)                              ← eval/synth/nb2_<변형>/validation_v4.csv (validate_replica.py)
  위치 축: 무작위 자리 / 원래 자리 검출률 (k = 1, 0.5)        ← eval/synth/stress_pos_site_<변형>_hd.csv (val·test 기증 결함, stress_pos.py)
출력: eval/out_nb2/variants.csv, 콘솔
"""
import os
import numpy as np
import pandas as pd

K = '/data/knhyun/KAMP'; O = f'{K}/eval/out_nb2'
V = [('yolov8s_scratch', 'YOLOv8s', '① 없음'), ('yolov8s', 'YOLOv8s', '② COCO'), ('yolov8s_pidray', 'YOLOv8s', '③ COCO→PIDray'),
     ('yolov8s_tpX', 'YOLOv8s', '② + 학습량 대조'), ('yolov8s_tpB', 'YOLOv8s', '② + 옮겨 심기 B'), ('yolov8s_tpC', 'YOLOv8s', '② + 옮겨 심기 C'),
     ('yolov8s_tpCC', 'YOLOv8s', '② + 옮겨 심기 C + 학습량 대조'), ('yolov8s_tpG', 'YOLOv8s', '② + 옮겨 심기 C + gVXR'),
     ('dfine_s_scratch', 'D-FINE-S', '① 없음'), ('dfine_s', 'D-FINE-S', '② COCO'), ('dfine_s_pidray', 'D-FINE-S', '③ COCO→PIDray'),
     ('dfine_s_tpB', 'D-FINE-S', '② + 옮겨 심기 B'), ('dfine_s_tpC', 'D-FINE-S', '② + 옮겨 심기 C'),
     ('dfine_s_tpCC', 'D-FINE-S', '② + 옮겨 심기 C + 학습량 대조'), ('dfine_s_tpG', 'D-FINE-S', '② + 옮겨 심기 C + gVXR'), ('', 'YOLOv3-SPP', '② COCO (416)')]
pm = pd.read_csv(f'{O}/per_machine.csv')
old = pd.read_csv(f'{K}/eval/out_v2/per_machine.csv'); old = old[old.run.str.startswith('v2/B_s') & (old.input == 'clean')]
ms = lambda x: f'{np.mean(x):.3f} ± {np.std(x, ddof=1):.3f}' if len(x) > 1 else (f'{x[0]:.3f}' if len(x) else '')
rows = []
for tag, model, cond in V:
    if tag:
        d = pm[pm.run.str.fullmatch(rf'nb2/{tag}_s\d')]
    else:
        d = old
    if not len(d):
        continue
    r = dict(model=model, cond=cond, n_seed=d.run.nunique())
    for sp, sc, col in [('test', 'all', 'test_AP'), ('test', '3호기', 'test_AP_m3'), ('val', 'all', 'val_AP')]:
        x = d[(d.split == sp) & (d.scope == sc)].sort_values('run').AP50.values; r[col] = ms(x)
    f4 = f'{K}/eval/synth/' + (f'nb2_{tag}/' if tag else '') + 'validation_v4.csv'
    if os.path.exists(f4):
        r['erased_hit25'] = pd.read_csv(f4, index_col=0).iloc[:, 0].get('erased_site_hit25', np.nan)
    fp = f'{K}/eval/synth/stress_pos_site' + (f'_{tag}' if tag else '') + '_hd.csv'
    if os.path.exists(fp):
        P = pd.read_csv(fp); P['hit'] = P.conf >= .25
        for kind, k, col in [('random', 1.0, 'pos_rand_k1'), ('random', .5, 'pos_rand_k05'), ('canon', 1.0, 'pos_canon_k1'), ('canon', .5, 'pos_canon_k05')]:
            g = P[(P.kind == kind) & (P.level == k)]
            r[col] = ms(g.groupby('model').hit.mean().values)
        g = P[(P.kind == 'random') & (P.level == 1.0)]
        for m in ['1호기', '2호기', '3호기']:
            r[f'pos_rand_k1_{m}'] = g[g.machine == m].hit.mean()
    rows.append(r)
T = pd.DataFrame(rows); T.to_csv(f'{O}/variants.csv', index=False)
pd.set_option('display.width', 260); pd.set_option('display.max_columns', 30)
print(T.round(3).to_string(index=False))
