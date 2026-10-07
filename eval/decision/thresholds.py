"""요구사항 ③ 판정 기준: 자동 불량(T_high) · 재검사(T_low ~ T_high) · 통과(< T_low), 제품 영역 후처리 효과
입력: eval/decision/dets_<변형>.csv, gts.csv (collect_dets.py), eval/synth/stress_pos_site_<변형>_hd.csv (위치 축, val·test 기증 결함)
규칙 (검출 단위, 위치는 중심거리 ≤ 8px 로 판정. 정답 박스 크기가 들쭉날쭉해서 IoU 대신 사용, §1.6):
  검출 신뢰도 ≥ T_high → 자동 불량,  T_low ≤ 신뢰도 < T_high → 재검사,  < T_low → 버림. 사진 판정 = 그 사진 검출 중 가장 높은 등급
  제품 영역 후처리: 검출 중심이 제품 영역(4px 확장) 밖이면 버림
임계값은 seed 마다 val 로만 정함:
  T_high = val 의 '결함 아닌 검출'(실제 영상의 짝 없는 검출 + 결함을 지운 영상의 모든 검출) 최고 신뢰도 바로 위 → val 에서 자동 불량 오판 0
  T_low  = 정책 R (실제 기준): val 실제 결함 전부를 잡는 최대값 (결함별 신뢰도의 최솟값)
           정책 P (위치 일반화): 정책 R 과 'val 위치 축 무작위 자리 k = 1 결함의 99%' 를 잡는 값 중 작은 것
평가: test 실제 결함(자동/재검사/놓침), test 실제 영상 오검출, test 결함 지운 영상(결함 없는 제품 대용) 사진 판정,
      test 위치 축 무작위 자리 k = 1 / 0.5 (자동/재검사/놓침), GT 없는 2,020장 사진 판정 비율(재검사 부담)
출력: eval/decision/thresholds.csv (seed별), summary.csv (변형별 평균), 콘솔
"""
import os, sys
import numpy as np
import pandas as pd

K = '/data/knhyun/KAMP'; D_ = f'{K}/eval/decision'
VARS = sys.argv[1:] or ['yolov8s', 'dfine_s']   # 합성 데이터는 평가 전용(팀 결정 2026-10-06) → 합성 학습 모델은 기본 대상에서 뺌
split = pd.read_csv(f'{K}/data/splits/split.csv').set_index('stem')
G = pd.read_csv(f'{D_}/gts.csv')
EPS = 1e-4


def grade(c, lo, hi):
    return np.where(c >= hi, 2, np.where(c >= lo, 1, 0))   # 2 자동 불량, 1 재검사, 0 통과/놓침


rows = []
for v in VARS:
    A = pd.read_csv(f'{D_}/dets_{v}.csv')
    for c in ['in_prod', 'gt_match']:                  # 결측 때문에 object 로 읽힘 → ~ 가 비트 반전(-2/-1)이 되지 않게 bool 로
        A[c] = A[c].map({True: True, False: False, 'True': True, 'False': False}).astype('boolean').fillna(False).astype(bool)
    S = pd.read_csv(f'{K}/eval/synth/stress_pos_site_{v}_hd.csv'); S = S[S.kind == 'random']; S['split'] = S.stem.map(split.split)
    for run, a in A.groupby('run'):
        for pp in [False, True]:                       # 제품 영역 후처리 없음 / 있음
            d = a.dropna(subset=['conf'])
            if pp:
                d = d[d.in_prod]
            d = d.assign(split=d.stem.map(split.split))
            real = d[d.set == 'real']
            # 결함별 신뢰도 (짝지어진 검출의 신뢰도, 없으면 0)
            gc = G.merge(real[real.gt_match][['stem', 'gt_id', 'conf']], on=['stem', 'gt_id'], how='left').fillna({'conf': 0.0})
            neg_val = pd.concat([real[(~real.gt_match) & (real.split == 'val')].conf, d[(d.set == 'erased') & (d.split == 'val')].conf])
            t_high = float(neg_val.max()) + EPS if len(neg_val) else 0.5
            t_low_R = float(gc[gc.split == 'val'].conf.min())
            sv = S[(S.model == run) & (S.split == 'val') & (S.level == 1.0)].conf
            t_low_P = min(t_low_R, float(np.quantile(sv, 0.01))) if len(sv) else t_low_R
            for pol, t_low in [('R', t_low_R), ('P', t_low_P)]:
                lo, hi = max(min(t_low, t_high), 1e-3), max(t_high, t_low)
                r = dict(variant=v, run=run, postproc=pp, policy=pol, T_low=lo, T_high=hi)
                gt = gc[gc.split == 'test']; g = grade(gt.conf.values, lo, hi)
                r.update(def_auto=(g == 2).mean(), def_reinsp=(g == 1).mean(), def_miss=(g == 0).mean())
                fp = real[(~real.gt_match) & (real.split == 'test')]
                n_img = (split.split == 'test').sum()
                r.update(fp_auto_per_img=(fp.conf >= hi).sum() / n_img, fp_reinsp_per_img=((fp.conf >= lo) & (fp.conf < hi)).sum() / n_img)
                e = a[(a.set == 'erased') & (a.stem.map(split.split) == 'test')]
                if pp:
                    e = e[e.in_prod.fillna(False).astype(bool) | e.conf.isna()]
                em = e.groupby('stem').conf.max().reindex(split[split.split == 'test'].index).fillna(0).values; ge = grade(em, lo, hi)
                r.update(neg_img_auto=(ge == 2).mean(), neg_img_reinsp=(ge == 1).mean(), neg_img_pass=(ge == 0).mean())
                for k in [1.0, 0.5]:
                    st = S[(S.model == run) & (S.split == 'test') & (S.level == k)].conf.values; gs = grade(st, lo, hi)
                    r.update({f'pos{k}_auto': (gs == 2).mean(), f'pos{k}_reinsp': (gs == 1).mean(), f'pos{k}_miss': (gs == 0).mean()})
                u = a[a.set == 'unlab']
                if pp:
                    u = u[u.in_prod.fillna(False).astype(bool) | u.conf.isna()]
                um = u.groupby('stem').conf.max().reindex(a[a.set == 'unlab'].stem.unique()).fillna(0).values; gu = grade(um, lo, hi)
                r.update(unl_auto=(gu == 2).mean(), unl_reinsp=(gu == 1).mean(), unl_pass=(gu == 0).mean())
                rows.append(r)
T = pd.DataFrame(rows); T.to_csv(f'{D_}/thresholds.csv', index=False)
num = [c for c in T.columns if c not in ('variant', 'run', 'postproc', 'policy')]
M = T.groupby(['variant', 'policy', 'postproc'], sort=False)[num].mean(); M.to_csv(f'{D_}/summary.csv')
pd.set_option('display.width', 280); pd.set_option('display.max_columns', 40)
print(M.round(3).to_string())
print('\nseed별 임계값'); print(T[T.postproc].pivot_table(index=['variant', 'run'], columns='policy', values=['T_low', 'T_high']).round(3).to_string())
