"""새 베이스라인(YOLOv8s, D-FINE-S) vs 기존 YOLOv3-SPP: 다시 돌린 실험 비교표 (모두 val·test, seed 3개)
입력: eval/out_nb2/summary.csv (nb2_eval.py), eval/marker_nb2/<모델>/trace_summary_inpaint.csv, eval/failure/per_defect*.csv, per_fp*.csv,
      eval/synth/[nb2_<모델>/]validation_summary.csv·validation_v4.csv, eval/synth/stress_v0_per_defect*.csv
출력: eval/out_nb2/compare_{trace,failure,synth,stress}.csv, eda/fig/n1_stress_models.png, 콘솔
"""
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False, 'axes.spines.top': False, 'axes.spines.right': False})

K = '/data/knhyun/KAMP'; O = f'{K}/eval/out_nb2'; os.makedirs(O, exist_ok=True)
MODELS = {'YOLOv3-SPP 416': '', 'YOLOv8s 640': 'yolov8s', 'D-FINE-S 640': 'dfine_s'}
COL = {'YOLOv3-SPP 416': '#8a8984', 'YOLOv8s 640': '#2a78d6', 'D-FINE-S 640': '#eb6834'}
MC = ['1호기', '2호기', '3호기']
have = {n: t for n, t in MODELS.items() if t == '' or os.path.exists(f'{K}/eval/failure/per_defect_{t}.csv')}
pd.set_option('display.width', 220)
split = pd.read_csv(f'{K}/data/splits/split.csv').set_index('stem')


def fit50(x, y):
    """P(검출) = 1 / (1 + exp(-(a + b·ln x))) → 50%, 90% 지점 (stress_v0_fig.py 와 같은 식)"""
    lx = np.log(np.clip(x, .2, None))
    nll = lambda p: -np.sum(y * -np.logaddexp(0, -(p[0] + p[1] * lx)) + (1 - y) * -np.logaddexp(0, (p[0] + p[1] * lx)))
    a, b = minimize(nll, [0, 1], method='Nelder-Mead').x
    return np.exp(-a / b), np.exp((np.log(9) - a) / b)


# 1. 마커 흔적 검사: 결함 없는 자리에 가짜 마커를 지운 흔적 → 반응률 (대조군 = 같은 자리 원본)
rows = []
for n, t in have.items():
    f = f'{K}/eval/marker_v2/trace_summary_inpaint.csv' if not t else f'{K}/eval/marker_nb2/{t}/trace_summary_inpaint.csv'
    if not os.path.exists(f):
        continue
    d = pd.read_csv(f); d = d[d.run.str.startswith('v2/B_s')] if not t else d
    for c, g in d.groupby('conf_thr'):
        rows.append(dict(model=n, conf_thr=c, control_hit=g.control_hit.mean(), traced_hit=g.traced_hit.mean(), n_trace=g.n_trace.sum()))
T = pd.DataFrame(rows); T.to_csv(f'{O}/compare_trace.csv', index=False)
print('■ 마커 흔적 검사 (가짜 링 450곳 × seed 3)'); print(T.pivot(index='conf_thr', columns='model', values='traced_hit').round(4).to_string() if len(T) else '없음')

# 2. 실패 조건: val·test 결함 350개 × seed 3
rows = []
for n, t in have.items():
    p = pd.read_csv(f'{K}/eval/failure/per_defect{"_" + t if t else ""}.csv')
    p = p[p.split.isin(['val', 'test'])]
    if not t:
        p = p[p.model.str.startswith('v2/B_s')]
    f = f'{K}/eval/failure/per_fp{"_" + t if t else ""}.csv'
    fp = pd.read_csv(f) if os.path.getsize(f) > 1 else pd.DataFrame(columns=['model', 'stem'])   # 오검출 0건이면 빈 파일
    fp = fp[fp.stem.map(split.split).isin(['val', 'test'])] if len(fp) else fp
    if not t and len(fp):
        fp = fp[fp.model.str.startswith('v2/B_s')]
    cond = {'전체': p.index == p.index, **{m: p.machine == m for m in MC},
            '3호기 CNR ≤ 8': (p.machine == '3호기') & (p.cnr <= 8), '3호기 CNR 8~10': (p.machine == '3호기') & (p.cnr > 8) & (p.cnr <= 10),
            '3호기 CNR > 10': (p.machine == '3호기') & (p.cnr > 10), '1·2호기 CNR ≤ 10': (p.machine != '3호기') & (p.cnr <= 10)}
    for c, k in cond.items():
        g = p[k]
        rows.append(dict(model=n, cond=c, n=len(g), miss25=(g.conf_at < .25).mean(), never=(g.conf_at < .01).mean(),
                         iou_bad_of_found=((g.conf_at >= .25) & (g.iou < .5)).sum() / max((g.conf_at >= .25).sum(), 1)))
    rows.append(dict(model=n, cond='오검출 (신뢰도 ≥ 0.25, 건수 / seed)', n=len(fp) / 3))
Fa = pd.DataFrame(rows); Fa.to_csv(f'{O}/compare_failure.csv', index=False)
print('\n■ 실패 조건: 놓침률 (신뢰도 < 0.25)'); print(Fa.pivot(index='cond', columns='model', values='miss25').round(3).to_string())
print('찾은 결함 중 IoU < 0.5 비율'); print(Fa.pivot(index='cond', columns='model', values='iou_bad_of_found').round(3).loc[['전체']].to_string())
print('오검출'); print(Fa[Fa.cond.str.startswith('오검출')][['model', 'n']].to_string(index=False))

# 3. 합성 검증 V3·V4: 옮겨 심기(T_cross)·매개변수(R1_cal)에 대한 모델 반응이 실제와 같은가
rows = []
for n, t in have.items():
    d = f'{K}/eval/synth' + (f'/nb2_{t}' if t else '')
    if not os.path.exists(f'{d}/validation_summary.csv'):
        continue
    S = pd.read_csv(f'{d}/validation_summary.csv').set_index('version'); V4 = pd.read_csv(f'{d}/validation_v4.csv', index_col=0).iloc[:, 0]
    D = pd.read_csv(f'{d}/replica_per_defect.csv'); D = D[D.stem.map(split.split).isin(['val', 'test'])]
    real = D[D.version == 'real'].set_index(['model', 'stem', 'obj'])
    for v in ['T_cross', 'R1_cal']:
        y = D[D.version == v].set_index(['model', 'stem', 'obj']).reindex(real.index)
        a = (real.machine == '3호기') & (real.cnr_real > 8) & (real.cnr_real <= 10)
        rows.append(dict(model=n, version=v, miss_real=(real.conf < .25).mean(), miss_syn=(y.conf < .25).mean(),
                         anchor_n=int(a.sum()), anchor_miss_real=(real.conf[a] < .25).mean(), anchor_miss_syn=(y.conf[a] < .25).mean(),
                         dconf=(y.conf - real.conf).mean(), conf_corr=np.corrcoef(real.conf, y.conf)[0, 1],
                         doffset=(y.offset - real.offset).mean(), null_hit10=V4.null_hit10, erased_site_hit10=(D[D.version == 'erased'].conf >= .1).mean()))
Sy = pd.DataFrame(rows); Sy.to_csv(f'{O}/compare_synth.csv', index=False)
print('\n■ 합성 검증 V3·V4 (val·test 결함)'); print(Sy.round(3).to_string(index=False))

# 4. 스트레스 곡선: 명목 대비 50%·90% 지점(밀도 축), 검출률 50% 가 되는 추가 잡음 σ
rows, curves = [], []
for n, t in have.items():
    f = f'{K}/eval/synth/stress_v0_per_defect{"_" + t if t else ""}.csv'
    if not os.path.exists(f):
        continue
    D = pd.read_csv(f); D['hit'] = D.conf >= .25
    parts = []
    for ax_, lv in [('N', 0.0), ('D_param', 1.0), ('D_trans', 1.0)]:   # 기준 대비: 잡음 0 / 배율 1 에서 잰 값
        ref = D[(D.axis == ax_) & (D.level == lv)].groupby(['stem', 'obj']).contrast.first().rename('c_ref')
        parts.append(D[D.axis == ax_].join(ref, on=['stem', 'obj']))
    D = pd.concat(parts)
    D['c_nom'] = np.where(D.axis == 'N', D.c_ref, D.c_ref * D.level)
    for m in MC + ['전체']:
        g0 = D if m == '전체' else D[D.machine == m]
        r = dict(model=n, machine=m)
        for ax_ in ['D_trans', 'D_param']:
            g = g0[g0.axis == ax_]; r[f'{ax_}_c50'], r[f'{ax_}_c90'] = fit50(g.c_nom.values, g.hit.values.astype(float))
        c = g0[g0.axis == 'N'].groupby('level').hit.mean()
        r['N_sigma50'] = float(np.interp(-.5, -c.values, c.index.values)) if c.min() < .5 else np.inf
        r['N_recall_at_sigma4'] = c.get(4.0, np.nan)
        rows.append(r)
        g = g0[g0.axis == 'D_trans'].assign(b=lambda x: pd.cut(x.c_nom, [0, 2, 4, 6, 8, 10, 12, 15, 20, 25, 30, 40, 60]))
        cc = g.groupby('b', observed=True).agg(x=('c_nom', 'median'), r=('hit', 'mean'), k=('hit', 'size')); cc = cc[cc.k >= 15]
        curves.append((n, m, 'D', cc.x.values, cc.r.values)); curves.append((n, m, 'N', c.index.values, c.values))
St = pd.DataFrame(rows); St.to_csv(f'{O}/compare_stress.csv', index=False)
print('\n■ 스트레스 곡선 (대비 단위 회색조, 옮겨 심기 기준)'); print(St.round(2).to_string(index=False))

if curves:
    fig, axs = plt.subplots(2, 3, figsize=(16, 8.4), sharey=True)
    for j, m in enumerate(MC):
        for n, mm, kind, x, y in curves:
            if mm != m:
                continue
            a = axs[0 if kind == 'D' else 1, j]
            a.plot(x, y, 'o-', color=COL[n], lw=2, ms=4, label=n)
        axs[0, j].set_title(m, weight='bold'); axs[0, j].set_xlabel('결함 대비 (회색조)'); axs[0, j].set_xlim(0, 45)
        axs[1, j].set_xlabel('추가한 흰 잡음 σ (회색조)')
        for a in axs[:, j]:
            a.grid(alpha=.3); a.set_ylim(0, 1.03)
    axs[0, 0].set_ylabel('검출률 (신뢰도 ≥ 0.25)\n결함을 옅게 (옮겨 심기)'); axs[1, 0].set_ylabel('검출률 (신뢰도 ≥ 0.25)\n잡음 추가')
    axs[0, 0].legend(fontsize=9, loc='lower right')
    fig.suptitle('스트레스 곡선: 모델별 비교 (val·test 결함 350개 × seed 3개, 학습 없음)', weight='bold')
    fig.tight_layout(); fig.savefig(f'{K}/eda/fig/n1_stress_models.png', dpi=120, bbox_inches='tight')
