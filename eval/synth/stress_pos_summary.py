"""위치 축 스트레스 요약 (stress_pos.py 결과 전부): 모델(변형)별 무작위 자리 vs 원래 자리 검출률, 빈 자리 반응, 대비 50% 지점, 자리 특성별 검출률
입력: eval/synth/stress_pos_site*.csv, stress_pos_fp*.csv
출력: eval/out_nb2/pos_summary.csv, pos_by_feature.csv, eda/fig/n2_stress_pos.png, 콘솔
"""
import glob, os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False, 'axes.spines.top': False, 'axes.spines.right': False})
K = '/data/knhyun/KAMP'; S = f'{K}/eval/synth'; O = f'{K}/eval/out_nb2'
NAME = {'': 'YOLOv3-SPP 416', 'yolov8s': 'YOLOv8s ② COCO', 'dfine_s': 'D-FINE-S ② COCO',
        'yolov8s_scratch': 'YOLOv8s ① 없음', 'yolov8s_pidray': 'YOLOv8s ③ PIDray', 'dfine_s_scratch': 'D-FINE-S ① 없음', 'dfine_s_pidray': 'D-FINE-S ③ PIDray',
        'yolov8s_tpB': 'YOLOv8s + 옮겨 심기 B', 'yolov8s_tpC': 'YOLOv8s + 옮겨 심기 C', 'yolov8s_tpX': 'YOLOv8s + 학습량 대조',
        'dfine_s_tpB': 'D-FINE-S + 옮겨 심기 B', 'dfine_s_tpC': 'D-FINE-S + 옮겨 심기 C'}
pd.set_option('display.width', 250)


def fit50(x, y):
    lx = np.log(np.clip(x, .2, None))
    nll = lambda p: -np.sum(y * -np.logaddexp(0, -(p[0] + p[1] * lx)) + (1 - y) * -np.logaddexp(0, (p[0] + p[1] * lx)))
    a, b = minimize(nll, [0, 1], method='Nelder-Mead').x
    return np.exp(-a / b) if b > 0 else np.nan


rows, feats, curves = [], [], {}
for f in sorted(glob.glob(f'{S}/stress_pos_site*.csv')):
    tag = os.path.basename(f)[len('stress_pos_site'):-4].lstrip('_'); hd = tag.endswith('hd'); base = tag[:-3] if tag.endswith('_hd') else ('' if tag == 'hd' else tag)
    n = NAME.get(base, base) + (' [val·test 기증]' if hd else '')
    D = pd.read_csv(f); D['hit'] = D.conf >= .25
    c1 = D[D.level == 1.0].groupby(['kind', 'stem', 'pass_', 'site']).contrast1.first().rename('c1')
    D = D.drop(columns='contrast1').join(c1, on=['kind', 'stem', 'pass_', 'site']); D['c_nom'] = D.level * D.c1
    F = pd.read_csv(f.replace('stress_pos_site', 'stress_pos_fp'))
    R, C = D[D.kind == 'random'], D[D.kind == 'canon']
    for scope, g in [('전체', None)] + [(m, m) for m in ['1호기', '2호기', '3호기']]:
        r_ = R if g is None else R[R.machine == g]; c_ = C if g is None else C[C.machine == g]
        per_seed = r_[r_.level == 1.0].groupby('model').hit.mean()
        rows.append(dict(model=n, scope=scope, n_sites=int((r_.level == 1.0).sum() / r_.model.nunique()),
                         rand_k1=r_[r_.level == 1.0].hit.mean(), rand_k1_sd=per_seed.std(), canon_k1=c_[c_.level == 1.0].hit.mean(),
                         rand_k05=r_[r_.level == .5].hit.mean(), canon_k05=c_[c_.level == .5].hit.mean(),
                         rand_k0=r_[r_.level == 0].hit.mean(), canon_k0=c_[c_.level == 0].hit.mean(),
                         rand_c50=fit50(r_[r_.level > 0].c_nom.values, r_[r_.level > 0].hit.values.astype(float)),
                         canon_c50=fit50(c_[c_.level > 0].c_nom.values, c_[c_.level > 0].hit.values.astype(float)),
                         fp_erased_per_img_k0=F[(F.kind == 'random') & (F.level == 0) & ((F.machine == g) if g else True)].fp_at_erased.mean(),
                         fp_other_per_img_k1=F[(F.kind == 'random') & (F.level == 1.0) & ((F.machine == g) if g else True)].fp_other.mean()))
    R1 = R[R.level == 1.0].copy()
    for col, bins in [('edge_dist', [0, 8, 16, 36, 200]), ('bg_rank', [0, .1, .25, .5, .75, 1.01]), ('dist_real', [20, 40, 80, 160, 1000])]:
        for b, g in R1.groupby(pd.cut(R1[col], bins, right=False), observed=True):
            feats.append(dict(model=n, feature=col, bin=str(b), n=len(g) // R1.model.nunique(), recall=g.hit.mean()))
    curves[n] = (R[R.level > 0].groupby('level').hit.mean(), C[C.level > 0].groupby('level').hit.mean())

T = pd.DataFrame(rows); T.to_csv(f'{O}/pos_summary.csv', index=False)
Fe = pd.DataFrame(feats); Fe.to_csv(f'{O}/pos_by_feature.csv', index=False)
cols = ['model', 'scope', 'n_sites', 'rand_k1', 'rand_k1_sd', 'canon_k1', 'rand_k05', 'canon_k05', 'rand_k0', 'canon_k0', 'rand_c50', 'canon_c50', 'fp_erased_per_img_k0', 'fp_other_per_img_k1']
print('■ 위치 축: 검출률 (신뢰도 ≥ 0.25). rand = 무작위 자리, canon = 원래 결함 자리, k = 광학 밀도 배율, c50 = 검출률 50% 명목 대비')
print(T[cols].round(3).to_string(index=False))
print('\n■ 무작위 자리, k = 1: 자리 특성별 검출률')
print(Fe.pivot_table(index=['feature', 'bin'], columns='model', values='recall', sort=False).round(3).to_string())

if curves:
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 5.2))
    cmap = plt.get_cmap('tab10')
    for i, (n, (r, c)) in enumerate(curves.items()):
        ax.plot(r.index, r.values, 'o-', color=cmap(i % 10), lw=2, ms=4, label=f'{n} (무작위 자리)')
        ax.plot(c.index, c.values, 's--', color=cmap(i % 10), lw=1, ms=3, alpha=.6)
    ax.set_xlabel('광학 밀도 배율 k (1 = 실제 결함 농도)'); ax.set_ylabel('검출률 (신뢰도 ≥ 0.25)'); ax.set_ylim(0, 1.03); ax.grid(alpha=.3)
    ax.set_title('위치 축 스트레스: 실선 = 제품 안 무작위 자리, 점선 = 원래 결함 자리', weight='bold', fontsize=11)
    ax.legend(fontsize=8, loc='lower right')
    fig.tight_layout(); fig.savefig(f'{K}/eda/fig/n2_stress_pos.png', dpi=120, bbox_inches='tight')
