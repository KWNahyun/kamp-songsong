"""미라벨 데이터 기술 통계 (모델 추론 없음)
데이터: 장비 원본 BMP 폴더 (eda/out/raw_inventory.csv, eda/extract.py 가 생성)
  GT 500장 / 미라벨 2,020장 / 제외 12장(튜토리얼용)
출력: eda/out/unlabeled_images.csv, eda/fig/u1~u3_*.png, 콘솔 요약
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

sys.argv = ['x']
exec(open('/data/knhyun/KAMP/eda/extract.py').read().split('# ─────────────────────────── 1.')[0])
FIG = '/data/knhyun/KAMP/eda/fig'

raw = pd.read_csv(f'{OUT}/raw_inventory.csv', parse_dates=['ts'])
dup = raw.groupby('stem').agg(n=('md5', 'size'), n_md5=('md5', 'nunique'))
print('파일', len(raw), '→ 고유 파일명', raw.stem.nunique(), '/ 고유 md5', raw.md5.nunique(),
      '/ 이름 중복', (dup.n > 1).sum(), '/ 이름 같고 내용 다름', (dup.n_md5 > 1).sum())
print('내용 다른 이름:', dup[dup.n_md5 > 1].index.tolist())

# 고유 이미지(md5) 기준. 세션은 GT·미라벨·제외 모두 포함한 전체 촬영 순서로 정의
u = raw.drop_duplicates('md5').sort_values(['machine', 'ts']).copy()
u['gap'] = u.groupby('machine').ts.diff().dt.total_seconds()
u['session'] = (u.gap.isna() | (u.gap > 600)).cumsum()
u['hour'] = u.ts.dt.hour
u['date'] = u.ts.dt.date

feats = []
for _, r in u.iterrows():
    gray, col, rgb = load_rgb(r.path)
    clean = remove_markers(gray, col)
    pm = product_mask(clean)
    lab, n_mk = ndi.label(col, structure=np.ones((3, 3)))
    # 장비 표시(마커) 중심이 제품 영역 안에 있는 개수 — 제품 밖 표시는 비정상 촬영 신호
    cms = [ndi.center_of_mass(col, lab, i) for i in range(1, n_mk + 1)]
    n_in = sum(bool(pm[int(round(y)), int(round(x))]) for y, x in cms)
    feats.append(dict(md5=r.md5, n_marker=n_mk, n_marker_in_product=n_in, marker_px=int(col.sum()),
                      prod_level=float(np.median(clean[pm])) if pm.any() else np.nan, prod_frac=float(pm.mean())))
u = u.merge(pd.DataFrame(feats), on='md5')
assert len(u) == raw.md5.nunique()
u.drop(columns=['path']).to_csv(f'{OUT}/unlabeled_images.csv', index=False)
gt, ul = u[u.use == 'GT'], u[u.use == '미라벨']
# 품질 플래그: 제품이 정상 크기로 찍혔는가(GT 최소 제품 면적 비율 기준) / 표시가 제품 안에 있는가
PF_MIN = gt.prod_frac.min()
u['q_product_ok'] = u.prod_frac.between(PF_MIN * 0.8, 0.5)
u['q_marker_in'] = u.n_marker_in_product > 0
u['quality'] = np.where(u.q_product_ok & u.q_marker_in, '정상', '비정상')
u.drop(columns=['path']).to_csv(f'{OUT}/unlabeled_images.csv', index=False)
gt, ul = u[u.use == 'GT'], u[u.use == '미라벨']
print(f'\n== 품질 (제품 면적 비율 기준 {PF_MIN * 0.8:.3f}~0.5, 표시 제품 안)')
print(pd.crosstab([u.use, u.w.astype(str) + 'x' + u.h.astype(str)], u.quality))
bad = ul[ul.quality == '비정상']
print('비정상 미라벨 날짜별:', bad.groupby(['machine', 'date']).size().to_dict())
print('비정상 사유: 제품 불량', (~bad.q_product_ok).sum(), '/ 표시 제품 밖', (~bad.q_marker_in).sum())

print('\n== 구분 × 장비'); print(pd.crosstab(u.machine, u.use, margins=True))
print('\n== 미라벨 해상도 × 장비'); print(pd.crosstab(ul.machine, ul.w.astype(str) + 'x' + ul.h.astype(str)))
print('\n== GT 해상도 × 장비'); print(pd.crosstab(gt.machine, gt.w.astype(str) + 'x' + gt.h.astype(str)))
d_gt, d_all = gt.groupby('machine').date.nunique(), u.groupby('machine').date.nunique()
print('\n== 촬영 날짜 수 (GT / 전체):', {m: f'{d_gt[m]}/{d_all[m]}' for m in d_all.index})
print('GT 가 없는 날짜의 미라벨 비율:', round((~ul.date.isin(set(gt.date))).mean(), 3))
s = u.groupby('session').agg(n=('stem', 'size'), n_gt=('use', lambda x: (x == 'GT').sum()), n_ul=('use', lambda x: (x == '미라벨').sum()))
print('\n== 세션', len(s), '| GT 포함', (s.n_gt > 0).sum(), '| GT 없는 세션', (s.n_gt == 0).sum())
print('GT 포함 세션 안의 미라벨:', int(s.n_ul[s.n_gt > 0].sum()), '| GT 없는 세션의 미라벨:', int(s.n_ul[s.n_gt == 0].sum()))
print('세션 크기 중앙값', s.n.median(), '최대', s.n.max())
big = s.n.idxmax(); bg = u[u.session == big]
print('최대 세션:', bg.machine.iloc[0], bg.ts.min(), '~', bg.ts.max(), '| 날짜별 최대:',
      u.groupby(['machine', 'date']).size().sort_values().tail(1).to_dict())
split = pd.read_csv('/data/knhyun/KAMP/data/splits/split.csv').set_index('stem')
gs = gt.assign(sp=gt.stem.map(split.split)).groupby('session').sp.agg(set)
inses = ul[ul.session.isin(gs.index)]
print('GT 세션에 섞인 미라벨의 해당 분할:', inses.session.map(lambda x: '/'.join(sorted(gs[x]))).value_counts().to_dict())
print('\n== 정시 촬영 비율: GT', round(gt.hour.isin([0, 4, 8, 12, 16, 20]).mean(), 3), '미라벨', round(ul.hour.isin([0, 4, 8, 12, 16, 20]).mean(), 3))
print('\n== 이미지당 마커 연결요소 수 비율'); print(pd.crosstab(u.use, u.n_marker.clip(upper=6), normalize='index').round(3))
print(pd.crosstab([u.use, u.machine], u.n_marker.clip(upper=6)))
print('\n== 제품 밝기 중앙값'); print(u[u.use != '제외(튜토리얼)'].groupby(['machine', 'use']).prod_level.describe()[['count', '25%', '50%', '75%', 'max']].round(1))

# ── 그림
SURF, INK2, GRID = '#fcfcfb', '#52514e', '#e4e3df'
MC = {'1호기': '#2a78d6', '2호기': '#eb6834', '3호기': '#1baf7a'}
plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False, 'figure.facecolor': SURF,
                     'axes.facecolor': SURF, 'savefig.facecolor': SURF, 'axes.grid': True, 'grid.color': GRID,
                     'axes.axisbelow': True, 'axes.spines.top': False, 'axes.spines.right': False,
                     'legend.frameon': False, 'axes.titleweight': 'bold', 'axes.titlesize': 12,
                     'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2})
MS = ['1호기', '2호기', '3호기']

fig, ax = plt.subplots(3, 1, figsize=(15, 7), sharex=True)
days = pd.date_range(u.ts.min().normalize(), u.ts.max().normalize())
for i, m in enumerate(MS):
    g = u[u.machine == m]
    cg = g[g.use == 'GT'].groupby(g.ts.dt.normalize()).size().reindex(days, fill_value=0)
    cu = g[g.use == '미라벨'].groupby(g.ts.dt.normalize()).size().reindex(days, fill_value=0)
    ax[i].bar(days, cu.values, color='#d4d3ce', width=0.8, label='미라벨')
    ax[i].bar(days, cg.values, bottom=cu.values, color=MC[m], width=0.8, label='GT')
    ax[i].set_ylabel(m); ax[i].legend(loc='upper right', fontsize=8)
ax[0].set_title('날짜별 이미지 수 — GT 는 일부 날짜에 몰려 있고, 미라벨은 전 기간에 분포')
fig.autofmt_xdate(); fig.tight_layout()
fig.savefig(f'{FIG}/u1_dates.png', dpi=130, bbox_inches='tight'); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(14, 4))
v = u[u.use != '제외(튜토리얼)']
res = v.assign(res=v.w.astype(str) + '×' + v.h.astype(str))
ct = pd.crosstab([res.machine, res.use], res.res)
cols = {'316×332': '#2a78d6', '352×332': '#eb6834', '412×332': '#e34948', '576×444': '#1baf7a'}
left = np.zeros(len(ct)); labels = [f'{m} {g}' for m, g in ct.index]
for c in ct.columns:
    ax[0].barh(labels, ct[c].values, left=left, color=cols[c], label=c, height=.6, edgecolor=SURF, linewidth=1.5)
    left += ct[c].values
ax[0].invert_yaxis(); ax[0].set_xlabel('이미지 수'); ax[0].set_title('해상도 구성 — 412×332 는 미라벨에만 존재')
ax[0].legend(fontsize=8, loc='lower right'); ax[0].grid(axis='y', visible=False)
for m in MS:
    ax[1].hist(ul[ul.machine == m].prod_level, bins=np.arange(90, 160, 1.5), histtype='step', lw=2, color=MC[m], label=f'{m} 미라벨')
    ax[1].hist(gt[gt.machine == m].prod_level, bins=np.arange(90, 160, 1.5), histtype='step', lw=1.2, ls='--', color=MC[m], label=f'{m} GT')
ax[1].set_xlabel('제품 영역 밝기 중앙값 (마커 제거 후)'); ax[1].set_ylabel('이미지 수')
ax[1].set_title('제품 밝기 분포 — GT(점선) vs 미라벨(실선)'); ax[1].legend(fontsize=7, ncol=2)
fig.savefig(f'{FIG}/u2_domain.png', dpi=130, bbox_inches='tight'); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 3.6)); w = 0.38
for i, (key, name) in enumerate([('GT', 'GT'), ('미라벨', '미라벨')]):
    vv = u[u.use == key].n_marker.clip(upper=6).value_counts(normalize=True).reindex(range(0, 7), fill_value=0)
    ax.bar(np.arange(7) + (i - .5) * w, vv.values, width=w - .02, color=['#2a78d6', '#d4d3ce'][i], label=name)
ax.set_xticks(range(7), ['0', '1', '2', '3', '4', '5', '6+']); ax.set_xlabel('이미지당 마커 연결요소 수 (장비 NG 표시, 참고용)')
ax.set_ylabel('이미지 비율'); ax.set_title('이미지당 장비 표시 개수 분포'); ax.legend(); ax.grid(axis='x', visible=False)
fig.savefig(f'{FIG}/u3_markers.png', dpi=130, bbox_inches='tight'); plt.close(fig)
print('\nfigures saved')
