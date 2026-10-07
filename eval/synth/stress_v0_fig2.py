"""스트레스 곡선 v0 최종 그림: 대비 축과 잡음 축을 각자의 단위로 (CNR 한 축으로 합치지 않음 — 실제 영상은 잡음이 거의 없어 CNR 분모가 의미 없음)
(a) 밀도 배율 축: 명목 대비(k × 원래 대비, 회색조) 대비 검출률, 장비별, 옮겨 심기(실선)·매개변수(점선), 빈 자리 배경 구조 대비(세로 점선)
(b) 잡음 축: 추가한 흰 잡음 σ 대비 검출률, 장비별
입력: stress_v0_per_defect_nom.csv (stress_v0_check.py) → eda/fig/s1_stress_cnr.png, eval/synth/stress_v0_limits.csv
"""
import numpy as np, pandas as pd, cv2, sys
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False, 'axes.spines.top': False, 'axes.spines.right': False})
K = '/data/knhyun/KAMP'; sys.argv = ['x']; sys.path.insert(0, f'{K}/eval/synth')
from stress_v0_fig import fit50  # noqa
exec(open(f'{K}/eval/synth/stress_v0.py').read().split('models = {')[0])   # measure, P
D = pd.read_csv(f'{K}/eval/synth/stress_v0_per_defect_nom.csv')
MC = {'1호기': '#2a78d6', '2호기': '#1baf7a', '3호기': '#eb6834'}
null = []
for stem in D.stem.unique():
    bg = cv2.imread(f'{K}/data/synth_val/erased/images/{stem}.png', 0)
    null += [dict(machine=r.machine, c0=measure(bg, r.px, r.py)[0]) for r in P[P.stem == stem].itertuples()]
N0 = pd.DataFrame(null).groupby('machine').c0.median()
rows = []
fig, ax = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
bins = np.array([0, 2, 4, 6, 8, 10, 12, 15, 20, 25, 30, 40, 60])
for m, col in MC.items():
    for k, ls in [('D_trans', '-'), ('D_param', '--')]:
        g = D[(D.axis == k) & (D.machine == m)]
        c = g.assign(b=pd.cut(g.c_nom, bins)).groupby('b', observed=True).agg(x=('c_nom', 'median'), r=('hit', 'mean'), n=('hit', 'size'))
        c = c[c.n >= 15]
        ax[0].plot(c.x, c.r, ls, marker='o', ms=4, color=col, lw=2, label=f'{m} {"옮겨 심기" if k == "D_trans" else "매개변수"}')
        c50, c90 = fit50(g.c_nom.values, g.hit.values.astype(float))
        rows.append(dict(machine=m, axis=k, unit='대비(회색조)', at50=c50, at90=c90, null_clutter=N0[m]))
    ax[0].axvline(N0[m], color=col, ls=':', lw=1.5)
    g = D[(D.axis == 'N') & (D.machine == m)]
    c = g.groupby('level').hit.mean()
    ax[1].plot(c.index, c.values, 'o-', color=col, lw=2, ms=5, label=m)
    lv = c.index.values; r = c.values   # 검출률이 50%, 90% 로 떨어지는 σ (선형 보간)
    f = lambda t: float(np.interp(-t, -r, lv)) if r.min() < t else np.nan
    rows.append(dict(machine=m, axis='N', unit='추가 잡음 σ', at50=f(.5), at90=f(.9), null_clutter=np.nan))
ax[0].set_xlabel('결함 대비 (배경 − 결함, 회색조)'); ax[0].set_ylabel('검출률 (신뢰도 ≥ 0.25)'); ax[0].set_xlim(0, 45)
ax[0].set_title('(a) 결함을 옅게: 밀도 배율 0.1~1.3\n점선 = 결함 없는 자리에서 잰 배경 구조 대비', weight='bold', fontsize=11)
ax[0].legend(fontsize=8, ncol=2, loc='lower right'); ax[0].grid(alpha=.3)
ax[1].set_xlabel('추가한 흰 잡음 σ (회색조)'); ax[1].set_title('(b) 잡음을 추가: 실제 결함 그대로', weight='bold', fontsize=11)
ax[1].legend(fontsize=9); ax[1].grid(alpha=.3); ax[1].set_ylim(0, 1.03)
fig.suptitle('스트레스 곡선 v0 (val·test 결함 350개 × 베이스라인 seed 3개, 학습 없음)', weight='bold')
fig.tight_layout(); fig.savefig(f'{K}/eda/fig/s1_stress_cnr.png', dpi=120, bbox_inches='tight')
S = pd.DataFrame(rows); S.to_csv(f'{K}/eval/synth/stress_v0_limits.csv', index=False); print(S.round(2).to_string(index=False))
# 실제 결함의 대비 분포 (관측 범위)
print(D[(D.axis == 'D_trans') & (D.level == 1.0)].groupby('machine').c_ref.describe(percentiles=[.05, .5]).round(1))
