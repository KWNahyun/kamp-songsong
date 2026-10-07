"""스트레스 곡선 v0 그림·요약: 측정 CNR 대비 검출률(신뢰도 ≥ 0.25, 중심 8px), 축·장비별, CNR50 (로지스틱 적합)
입력: eval/synth/stress_v0_per_defect.csv, stress_v0_fp.csv → 출력: eda/fig/s1_stress_cnr.png, eval/synth/stress_v0_summary.csv
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False, 'axes.spines.top': False, 'axes.spines.right': False})

K = '/data/knhyun/KAMP'
D = pd.read_csv(f'{K}/eval/synth/stress_v0_per_defect.csv'); F = pd.read_csv(f'{K}/eval/synth/stress_v0_fp.csv')
D['hit'] = D.conf >= 0.25
OBS = (5.1, 35.9)   # 실제 GT 결함의 CNR 범위 (§3.6)
AX = {'N': ('실제 결함 + 잡음 추가', '#2a78d6'), 'D_trans': ('옮겨 심기, 밀도 배율', '#1baf7a'), 'D_param': ('매개변수 합성, 밀도 배율', '#eb6834')}
MC = ['1호기', '2호기', '3호기']


def fit50(x, y):
    """P(검출) = 1 / (1 + exp(-(a + b·ln CNR))) → 50%, 90% 지점"""
    lx = np.log(np.clip(x, .2, None))
    nll = lambda p: -np.sum(y * -np.logaddexp(0, -(p[0] + p[1] * lx)) + (1 - y) * -np.logaddexp(0, (p[0] + p[1] * lx)))
    a, b = minimize(nll, [0, 1], method='Nelder-Mead').x
    return np.exp(-a / b), np.exp((np.log(9) - a) / b)


rows = []
fig, ax = plt.subplots(1, 4, figsize=(20, 4.6), sharey=True)
bins = np.array([0, 1, 2, 3, 4, 5, 6, 8, 10, 13, 17, 22, 30, 45])
for i, m in enumerate(MC + ['전체']):
    a = ax[i]
    for k, (lab, col) in AX.items():
        g = D[(D.axis == k) & ((D.machine == m) if m != '전체' else True)]
        g = g.assign(b=pd.cut(g.cnr, bins))
        c = g.groupby('b', observed=True).agg(x=('cnr', 'median'), r=('hit', 'mean'), n=('hit', 'size'))
        c = c[c.n >= 15]
        a.plot(c.x, c.r, 'o-', color=col, label=lab, lw=2, ms=5)
        c50, c90 = fit50(g.cnr.values, g.hit.values.astype(float))
        rows.append(dict(machine=m, axis=k, CNR50=c50, CNR90=c90, n=len(g)))
    a.axvspan(*OBS, color='#e4e3df', alpha=.5, lw=0)
    a.text(OBS[0] * 1.05, .04, '실제 결함의 CNR 범위', fontsize=8.5, color='#52514e')
    a.set_xscale('log'); a.set_xlim(.4, 45); a.set_ylim(0, 1.03); a.set_title(m, weight='bold'); a.set_xlabel('측정 CNR (결함 대비 / 잡음)')
    a.grid(alpha=.3)
ax[0].set_ylabel('검출률 (신뢰도 ≥ 0.25)'); ax[0].legend(fontsize=9, loc='upper left')
fig.suptitle('스트레스 곡선 v0: 실제 결함 자리에서 대비만 낮췄을 때의 검출률 (val·test 결함 350개 × 베이스라인 seed 3개)', weight='bold')
fig.tight_layout(); fig.savefig(f'{K}/eda/fig/s1_stress_cnr.png', dpi=120, bbox_inches='tight'); plt.close(fig)
S = pd.DataFrame(rows); S.to_csv(f'{K}/eval/synth/stress_v0_summary.csv', index=False)
pd.set_option('display.width', 200)
print(S.pivot(index='machine', columns='axis', values='CNR50').round(2)); print(S.pivot(index='machine', columns='axis', values='CNR90').round(2))
print('이미지당 오검출(신뢰도 ≥ 0.25):'); print(F.groupby(['axis', 'level']).n_fp.mean().unstack(0).round(3))
