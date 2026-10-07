"""데이터 구성·분할 리포트 그림 → eda/fig/d1~d3_*.png"""
import os, sys
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

K = '/data/knhyun/KAMP'
FIG = f'{K}/eda/fig'
SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False, 'figure.facecolor': SURF,
                     'axes.facecolor': SURF, 'savefig.facecolor': SURF, 'axes.grid': True, 'grid.color': GRID,
                     'axes.axisbelow': True, 'axes.spines.top': False, 'axes.spines.right': False,
                     'legend.frameon': False, 'axes.titleweight': 'bold', 'axes.titlesize': 12,
                     'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'font.size': 10})
MS = ['1호기', '2호기', '3호기']

split = pd.read_csv(f'{K}/data/splits/split.csv')
split['ts'] = pd.to_datetime(split.stem.str[4:19], format='%Y%m%d_%H%M%S')
roles = pd.read_csv(f'{K}/data/splits/unlabeled_roles.csv')
roles['ts'] = pd.to_datetime(roles.stem.str[4:19], format='%Y%m%d_%H%M%S')

# ═══ D1. 날짜별 배치 타임라인 ═══
GTC = {'train': '#2a78d6', 'val': '#eda100', 'test': '#e34948'}
ULC = {'PL학습': ('#c9c8c3', 'o', 'GT 없음: 학습 가능'), 'PL평가': ('#8a8984', 'o', 'GT 없음: 평가용(처음 보는 날짜)'),
       '검증그룹(제외)': ('#e34948', 'x', 'GT 없음: val·test와 같은 날 → 제외'), '이상': ('#0b0b0b', '^', 'GT 없음: 이상 이미지')}
fig, ax = plt.subplots(3, 1, figsize=(16, 8.2), sharex=True)
rng = np.random.default_rng(0)
for i, m in enumerate(MS):
    a = ax[i]
    u = roles[roles.machine == m]
    for role, (c, mk, lab) in ULC.items():
        x = u[u.role == role]
        a.scatter(x.ts, 0.35 + rng.uniform(-.12, .12, len(x)), s=10 if mk != 'x' else 14, color=c, marker=mk, lw=1 if mk == 'x' else 0, label=lab)
    g = split[split.machine == m]
    for sp_, c in GTC.items():
        x = g[g.split == sp_]
        a.scatter(x.ts, 1.0 + rng.uniform(-.12, .12, len(x)), s=16, color=c, lw=0, label=f'GT 있음: {sp_}')
    a.set_yticks([0.35, 1.0], ['GT 없음', 'GT 있음']); a.set_ylim(0, 1.3); a.set_ylabel(m, rotation=0, labelpad=25, fontsize=11)
    a.grid(axis='y', visible=False)
ax[0].set_title('장비·날짜별 데이터 배치 — 같은 장비·같은 날의 사진은 한 곳에만 (색 = 분할, × = 누수 방지로 제외)')
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.04), fontsize=9, markerscale=1.8)
fig.autofmt_xdate(); fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(f'{FIG}/d1_split_timeline.png', dpi=130, bbox_inches='tight'); plt.close(fig)

# ═══ D2. 실제 사진 비교: 같은 세션 / 같은 날 다른 세션 / 다른 날 ═══
img = pd.read_csv(f'{K}/eda/out/images.csv')
img = img.merge(split[['stem', 'session', 'split']], on='stem')
img['date'] = img.stem.str[4:12]
dh = {s: np.array([c == '1' for c in h]) for s, h in zip(img.stem, img.dhash)}
dist = lambda a, b: int((dh[a] != dh[b]).sum())
# 3호기, 3개짜리 시편에서 세션 하나 고르기
g3 = img[(img.machine == '3호기') & (img.n_obj == 3)].sort_values('stem')
# 세션이 2개 이상 있는 날짜에서, 3장 이상인 세션을 기준으로
nses = g3.groupby('date').session.nunique()
cand = g3[g3.date.isin(nses[nses >= 2].index)]
s0 = cand.groupby('session').filter(lambda x: len(x) >= 3).session.iloc[0]
ses = g3[g3.session == s0].stem.tolist()[:3]
d0 = g3[g3.stem == ses[0]].date.iloc[0]
day = g3[(g3.date == d0) & (g3.session != s0)].stem.iloc[0]
other = g3[g3.date != d0].stem.iloc[len(g3) // 2]
picks = [(ses[0], '기준 사진'), (ses[1], '같은 세션 (몇 초 뒤)'), (ses[2], '같은 세션 (몇 초 뒤)'),
         (day, '같은 날, 다른 세션 (몇 시간 뒤)'), (other, '다른 날')]
fig, ax = plt.subplots(1, 5, figsize=(17, 3.9))
for a, (s, lab) in zip(ax, picks):
    im = cv2.imread(f'{K}/data/clean/images/{s}.png', 0)
    a.imshow(im, cmap='gray', vmin=60, vmax=215); a.axis('off')
    t = f'{s[4:12]} {s[13:15]}:{s[15:17]}:{s[17:19]}'
    extra = '' if s == ses[0] else f'\n기준과의 차이값 {dist(ses[0], s)}'
    a.set_title(f'{lab}\n{t}{extra}', fontsize=9)
fig.suptitle('3호기 GT 사진 예시 (마커 제거본) — 모두 같은 테스트피스를 반복 촬영. 개별 쌍의 차이값은 들쭉날쭉하므로 경향은 분포(D3)로 판단', weight='bold', y=1.04, fontsize=11)
fig.savefig(f'{FIG}/d2_similarity_examples.png', dpi=130, bbox_inches='tight'); plt.close(fig)

# ═══ D3. 비슷함 분포 + 분할 전후 겹침 ═══
H = np.stack([dh[s] for s in img.stem]); m, d, se, W = img.machine.values, img.date.values, img.session.values, img.W.values
D = (H[:, None] != H[None]).sum(-1); iu = np.triu_indices(len(img), 1)
k = (m[:, None] == m[None])[iu] & (W[:, None] == W[None])[iu]
cat = np.where((se[:, None] == se[None])[iu], '같은 세션', np.where((d[:, None] == d[None])[iu], '같은 날\n다른 세션', '다른 날'))
fig, a0 = plt.subplots(figsize=(8.5, 4.2)); ax = [a0]
cats = ['같은 세션', '같은 날\n다른 세션', '다른 날']; cols = ['#e34948', '#eda100', '#2a78d6']
data = [D[iu][k & (cat == c)] for c in cats]
bp = ax[0].boxplot(data, widths=.5, patch_artist=True, showfliers=False, medianprops=dict(color=INK))
for p, c in zip(bp['boxes'], cols): p.set_facecolor(c); p.set_alpha(.6)
for i, x in enumerate(data):
    ax[0].text(i + 1.3, np.median(x), f'중앙값 {np.median(x):.0f}\n({len(x):,}쌍)', va='center', fontsize=9, color=INK2)
ax[0].set_xticks([1, 2, 3], cats); ax[0].set_ylabel('사진 간 차이값 (dHash 거리, 작을수록 비슷)')
ax[0].set_title('GT 사진 쌍의 비슷함 (같은 장비끼리, 마커 제거본)'); ax[0].grid(axis='x', visible=False)
fig.savefig(f'{FIG}/d3_leakage_evidence.png', dpi=130, bbox_inches='tight'); plt.close(fig)

# ═══ D4. val 사진과 같은 날 찍혀 학습에서 제외된 GT 없는 사진 (3호기 2020-09-22) ═══
vg = split[(split.machine == '3호기') & (split.date.astype(str) == '20200922')].stem.tolist()
ex = roles[(roles.machine == '3호기') & (roles.stem.str[4:12] == '20200922')].sort_values('stem')
ex = ex.iloc[np.linspace(0, len(ex) - 1, 3).astype(int)]
items = [(f'{K}/data/clean/images/{s}.png', f'GT 있음 · val\n{s[13:15]}:{s[15:17]}:{s[17:19]}') for s in vg[:2]] + \
        [(f'{K}/data/unlabeled/images/{n}.png', f'GT 없음 · 학습에서 제외\n{n[13:15]}:{n[15:17]}:{n[17:19]}') for n in ex['name']]
fig, ax = plt.subplots(1, 5, figsize=(17, 3.6))
for a, (p_, lab) in zip(ax, items):
    a.imshow(cv2.imread(p_, 0), cmap='gray', vmin=60, vmax=215); a.axis('off'); a.set_title(lab, fontsize=10)
fig.suptitle('3호기 2020-09-22: val 사진 2장과 같은 날 찍힌 GT 없는 사진 → 같은 시편·같은 장비·같은 날이라 학습에서 제외', weight='bold', y=1.05, fontsize=11)
fig.savefig(f'{FIG}/d4_same_day_excluded.png', dpi=130, bbox_inches='tight'); plt.close(fig)

# ═══ D5. 이상 이미지 유형 (원본, 마커 포함) ═══
sys.argv = ['x']
exec(open(f'{K}/eda/extract.py').read().split('# ─────────────────────────── 1.')[0])
raw = pd.read_csv(f'{K}/eda/out/raw_inventory.csv').drop_duplicates('md5').set_index('md5')
ab = roles[roles.role == '이상']
pick = [('002_20200727_110524(7)', '제품이 화면 밖으로 잘림\n(1호기 7/27, 104장)'), ('002_20200727_115526(5)', '제품 없음\n(1호기 7/27)'),
        ('002_20200904_230042(8)', '결함 대신 "T" 표시\n(2호기 9/4, 2장)'), ('002_20200713_202507(0)', '표시가 제품 밖\n(1호기 7/13, 1장)')]
fig, ax = plt.subplots(1, 4, figsize=(15, 3.8))
for a, (st, lab) in zip(ax, pick):
    r = ab[ab.stem == st].iloc[0]
    _, _, rgb = load_rgb(raw.path[r.md5])
    a.imshow(rgb.astype(np.uint8)); a.axis('off'); a.set_title(lab, fontsize=10)
fig.suptitle('GT 없는 데이터의 이상 이미지 107장 (원본, 장비 표시 포함) — 학습 제외, 헛검출 점검용', weight='bold', y=1.05, fontsize=11)
fig.savefig(f'{FIG}/d5_abnormal_examples.png', dpi=130, bbox_inches='tight'); plt.close(fig)
print('saved d1~d5')
