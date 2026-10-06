"""스트레스 곡선 v0 점검: 대비↓ vs 잡음↑ 비대칭이 측정 편향 때문인지, 절대 대비 때문인지
- 명목 CNR: D 축은 k × (k=1 에서 잰 대비) / 잡음, N 축은 (잡음 0 에서 잰 대비) / 잡음  → 낮은 대비에서 최솟값 탐색이 대비를 부풀리는 편향 제거
- 빈 자리 측정: 결함을 지운 배경(k=0)에서 같은 측정을 하면 대비가 얼마로 나오는지 (편향 크기)
출력: eval/synth/stress_v0_check.csv, 콘솔
"""
import sys
import numpy as np, pandas as pd, cv2
K = '/data/knhyun/KAMP'
sys.argv = ['x']
sys.path.insert(0, f'{K}/eval/synth')
src = open(f'{K}/eval/synth/stress_v0.py').read().split('models = {')[0]   # measure·P 만 가져옴 (추론 없음)
exec(src)
from stress_v0_fig import fit50  # noqa  (그림도 다시 그려짐)

D = pd.read_csv(f'{K}/eval/synth/stress_v0_per_defect.csv'); D['hit'] = D.conf >= .25
key = ['stem', 'obj']
ref = {}
for ax, lv in [('N', 0.0), ('D_param', 1.0), ('D_trans', 1.0)]:
    ref[ax] = D[(D.axis == ax) & (D.level == lv)].groupby(key).contrast.first().rename('c_ref')
out = []
for ax, g in D.groupby('axis'):
    g = g.join(ref[ax], on=key)
    g['c_nom'] = g.c_ref * (g.level if ax != 'N' else 1.0)
    out.append(g)
D = pd.concat(out); D['cnr_nom'] = D.c_nom / D.noise.clip(lower=.5)

# 편향: 결함 없는 배경(지운 자리)에서 잰 '대비'
null = []
for stem in D.stem.unique():
    bg = cv2.imread(f'{K}/data/synth_val/erased/images/{stem}.png', 0)
    for r in P[P.stem == stem].itertuples():
        c, s = measure(bg, r.px, r.py); null.append(dict(stem=stem, obj=r.obj, machine=r.machine, c0=c, s0=s))
N0 = pd.DataFrame(null)
print('빈 자리 측정 대비 (중앙값, 장비별):'); print(N0.groupby('machine')[['c0', 's0']].median().round(2)); print('빈 자리 측정 CNR 중앙값', round(float((N0.c0 / N0.s0.clip(lower=.5)).median()), 2))

rows = []
for m in ['1호기', '2호기', '3호기', '전체']:
    for ax in ['N', 'D_trans', 'D_param']:
        g = D[(D.axis == ax) & ((D.machine == m) if m != '전체' else True)]
        a50, a90 = fit50(g.cnr.values, g.hit.values.astype(float)); n50, n90 = fit50(g.cnr_nom.values, g.hit.values.astype(float))
        c50, c90 = fit50(g.c_nom.values, g.hit.values.astype(float))
        rows.append(dict(machine=m, axis=ax, CNR50_meas=a50, CNR50_nom=n50, CNR90_nom=n90, contrast50=c50))
S = pd.DataFrame(rows); S.to_csv(f'{K}/eval/synth/stress_v0_check.csv', index=False)
pd.set_option('display.width', 200); print(S.round(2).to_string(index=False))

# 같은 명목 CNR 구간에서 축별 검출률 (절대 대비 효과)
D['cb'] = pd.cut(D.cnr_nom, [0, 2, 3, 4, 5, 6, 8, 10, 14, 50])
print(D.groupby(['cb', 'axis'], observed=True).agg(r=('hit', 'mean'), c=('c_nom', 'median'), n=('hit', 'size')).round(2).unstack('axis').to_string())
D.to_csv(f'{K}/eval/synth/stress_v0_per_defect_nom.csv', index=False)
