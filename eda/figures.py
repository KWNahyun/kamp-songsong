"""EDA 리포트 그림 생성 — eda/out/*.csv → eda/fig/*.png"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

sys.argv = ['x']
_src = open(os.path.join(os.path.dirname(__file__), 'extract.py')).read()
exec(_src.split('# ─────────────────────────── 1.')[0])   # 경로/헬퍼 함수 재사용

FIG = '/data/knhyun/KAMP/eda/fig'
os.makedirs(FIG, exist_ok=True)

# ── 스타일 (reference palette: 범주형 슬롯 1~3, 중립 잉크, 옅은 grid)
SURF, INK, INK2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8984', '#e4e3df'
MC = {'1호기': '#2a78d6', '2호기': '#eb6834', '3호기': '#1baf7a'}
SPLIT_C = {'train': '#2a78d6', 'val': '#eb6834', 'test': '#1baf7a'}
plt.rcParams.update({
    'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False,
    'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF,
    'axes.edgecolor': MUTED, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
    'text.color': INK, 'axes.titlecolor': INK, 'axes.titlesize': 12, 'axes.titleweight': 'bold',
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.axisbelow': True,
    'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
    'font.size': 10, 'figure.dpi': 110,
})
MACHINES = ['1호기', '2호기', '3호기']

raw = pd.read_csv(f'{OUT}/raw_inventory.csv', parse_dates=['ts'])
img = pd.read_csv(f'{OUT}/images.csv', parse_dates=['ts'])
obj = pd.read_csv(f'{OUT}/objects.csv')
mk = pd.read_csv(f'{OUT}/markers.csv')
obj['strip'] = obj.groupby('stem').cy.rank(method='first').astype(int)
stem2rawpath = raw.groupby('stem').path.first().to_dict()
img_path = lambda s, *_: stem2rawpath[s]   # 데이터 출처: 장비 원본 BMP 폴더


def save(fig, name):
    fig.savefig(f'{FIG}/{name}.png', bbox_inches='tight', dpi=130)
    plt.close(fig)


# ═══ F1. 데이터 구성: 원본 풀 vs 라벨링, 장비×날짜 타임라인, 촬영 시각 ═══
rawu = raw.drop_duplicates('md5').copy()
rawu['labeled'] = rawu.is_gt
fig, ax = plt.subplots(1, 3, figsize=(16, 4.2), gridspec_kw={'width_ratios': [1, 2.2, 1.2]})
cats = [('GT (공식 라벨)', 'GT', '#2a78d6'), ('미라벨', '미라벨', '#d4d3ce'), ('제외 (튜토리얼)', '제외(튜토리얼)', '#eda100')]
left = np.zeros(3)
for name, key, c in cats:
    v = np.array([((rawu.machine == m) & (rawu.use == key)).sum() for m in MACHINES])
    ax[0].barh(MACHINES, v, left=left, color=c, label=name, height=0.6, edgecolor=SURF, linewidth=2)
    left += v
for i, t in enumerate(left):
    ax[0].text(t + 15, i, f'{int(t)}', va='center', color=INK2)
ax[0].invert_yaxis(); ax[0].set_xlabel('이미지 수 (md5 중복 제거)')
ax[0].set_title('장비 원본 이미지: GT / 미라벨 구성'); ax[0].legend(loc='upper center', bbox_to_anchor=(0.45, -0.22), fontsize=8, ncol=1)
ax[0].grid(axis='y', visible=False)

for i, m in enumerate(MACHINES):
    r = rawu[rawu.machine == m]
    ax[1].scatter(r.ts, np.full(len(r), i) + np.random.default_rng(0).uniform(-.25, .25, len(r)),
                  s=4, color='#c9c8c3', lw=0)
    l = r[r.labeled]
    ax[1].scatter(l.ts, np.full(len(l), i) + np.random.default_rng(1).uniform(-.25, .25, len(l)),
                  s=10, color=MC[m], lw=0)
ax[1].set_yticks(range(3), MACHINES); ax[1].invert_yaxis()
ax[1].set_title('촬영 시점 (회색 = 미라벨·제외, 색 = GT)'); ax[1].grid(axis='y', visible=False)
fig.autofmt_xdate()

hr = rawu.ts.dt.hour.value_counts().reindex(range(24), fill_value=0)
hl = img.ts.dt.hour.value_counts().reindex(range(24), fill_value=0)
ax[2].bar(hr.index, hr.values, color='#d4d3ce', width=0.8, label='전체')
ax[2].bar(hl.index, hl.values, color='#2a78d6', width=0.8, label='GT')
ax[2].set_xticks([0, 4, 8, 12, 16, 20]); ax[2].set_xlabel('촬영 시각 (시)')
ax[2].set_title('4시간 주기 정기 점검 패턴'); ax[2].legend(fontsize=8)
save(fig, 'f1_inventory')

# ═══ F2. 장비별 대표 이미지 (원본 vs 마커 제거) ═══
fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
for j, m in enumerate(MACHINES):
    s = img[(img.machine == m) & (img.n_obj == 3) & img.labeled_in_400].stem.iloc[3]
    gray, col, rgb = load_rgb(img_path(s, True)); clean = remove_markers(gray, col)
    ax[0, j].imshow(rgb.astype(np.uint8)); ax[1, j].imshow(clean, cmap='gray', vmin=0, vmax=255)
    for b in load_labels(s):
        H, W = gray.shape
        ax[1, j].add_patch(Rectangle(((b[1] - b[3] / 2) * W - 3, (b[2] - b[4] / 2) * H - 3),
                                     b[3] * W + 6, b[4] * H + 6, fill=False, ec='#e34948', lw=1.2))
    ax[0, j].set_title(f'{m} · {gray.shape[1]}×{gray.shape[0]} · 원본 (장비 마커 포함)', fontsize=10)
    ax[1, j].set_title(f'{m} · 마커 제거(inpaint) + GT 박스', fontsize=10)
    for a in ax[:, j]: a.axis('off')
save(fig, 'f2_samples')

# ═══ F3. 마커 shortcut 증거 ═══
fig = plt.figure(figsize=(16, 4.4))
gs = fig.add_gridspec(1, 4, width_ratios=[2.2, 1, 1, 1.1])
sub = gs[0].subgridspec(2, 4, hspace=0.25, wspace=0.08)
pick = obj[obj.labeled_in_400].groupby('machine').nth([0, 40]).head(4)
for k, (_, o) in enumerate(pick.iterrows()):
    gray, col, rgb = load_rgb(img_path(o.stem, True)); clean = remove_markers(gray, col)
    x, y, r = int(o.cx * o.W), int(o.cy * o.H), 15
    a0 = fig.add_subplot(sub[0, k]); a1 = fig.add_subplot(sub[1, k])
    a0.imshow(rgb[max(y - r, 0):y + r, max(x - r, 0):x + r].astype(np.uint8), interpolation='nearest')
    a1.imshow(clean[max(y - r, 0):y + r, max(x - r, 0):x + r], cmap='gray', vmin=0, vmax=255, interpolation='nearest')
    for a in (a0, a1): a.set_xticks([]); a.set_yticks([]); a.grid(False)
    if k == 0: a0.set_ylabel('원본', fontsize=9); a1.set_ylabel('마커 제거', fontsize=9)
fig.text(0.02, 0.97, '결함 주변 30×30px 확대', fontsize=12, weight='bold')

a = fig.add_subplot(gs[1])
off = []
m1 = mk[mk.n_gt_inside == 1]
for _, r in m1.iterrows():
    c = obj[(obj.stem == r.stem) & (obj.cx * obj.W >= r.x0) & (obj.cx * obj.W <= r.x0 + r.w)
            & (obj.cy * obj.H >= r.y0) & (obj.cy * obj.H <= r.y0 + r.h)].iloc[0]
    off.append((r.x0 + r.w / 2 - c.cx * c.W, r.y0 + r.h / 2 - c.cy * c.H))
off = np.array(off)
a.hist2d(off[:, 0], off[:, 1], bins=np.arange(-4.25, 4.5, 0.5), cmap='Blues')
a.set_title('마커 중심 − GT 중심 (px)'); a.set_xlabel('dx'); a.set_ylabel('dy'); a.grid(False)
a.text(0.03, 0.95, f'{(np.hypot(*off.T) <= 2).mean():.1%} ≤ 2px', transform=a.transAxes, va='top', color=INK2)

a = fig.add_subplot(gs[2])
ct = pd.crosstab(img.n_marker_cc, img.n_obj)
a.imshow(ct.values, cmap='Blues'); a.grid(False)
for (i, j), v in np.ndenumerate(ct.values):
    a.text(j, i, v, ha='center', va='center', color='white' if v > 150 else INK)
a.set_xticks(range(ct.shape[1]), ct.columns); a.set_yticks(range(ct.shape[0]), ct.index)
a.set_xlabel('이미지당 GT 개수'); a.set_ylabel('이미지당 마커 개수'); a.set_title('마커 수 ↔ GT 수')

a = fig.add_subplot(gs[3])
cc = mk.color.str.split('+').explode().value_counts()
a.barh(cc.index, cc.values, color='#8a8984', height=0.6)
a.invert_yaxis(); a.set_title('마커 색상 구성 (연결요소 기준)'); a.grid(axis='y', visible=False)
for i, v in enumerate(cc.values): a.text(v + 10, i, v, va='center', color=INK2, fontsize=9)
fig.subplots_adjust(top=0.85)
save(fig, 'f3_marker_shortcut')

# ═══ F4. 결함 크기: 라벨 박스 vs 실제 결함 코어, 모델 입력(416) 기준 ═══
fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
for m in MACHINES:
    o = obj[obj.machine == m]
    ax[0].scatter(o.w_px + np.random.default_rng(0).uniform(-.3, .3, len(o)),
                  o.h_px + np.random.default_rng(1).uniform(-.3, .3, len(o)), s=9, color=MC[m], alpha=.5, lw=0, label=m)
ax[0].set_xlabel('GT 박스 너비 (px, 원본 해상도)'); ax[0].set_ylabel('GT 박스 높이 (px)')
ax[0].set_title('라벨 박스 크기 (장비별)'); ax[0].legend(markerscale=2)
bins = np.arange(1.5, 16.5, 1)
for m in MACHINES:
    ax[1].hist(obj[obj.machine == m].core_px, bins=bins, histtype='step', lw=2, color=MC[m], label=m)
ax[1].set_xlabel('결함 코어 면적 (px, FWHM 기준)'); ax[1].set_ylabel('객체 수')
ax[1].set_title('실제 결함 크기: 대부분 2×2 = 4px'); ax[1].legend()
obj['scale416'] = 416 / obj[['W', 'H']].max(1)
obj['side416'] = np.sqrt(obj.area_px) * obj.scale416
obj['core416'] = np.sqrt(obj.core_px) * obj.scale416
data = [obj[obj.machine == m].side416 for m in MACHINES] + [obj[obj.machine == m].core416 for m in MACHINES]
bp = ax[2].boxplot(data, positions=[1, 2, 3, 5, 6, 7], widths=.6, patch_artist=True, showfliers=False,
                   medianprops=dict(color=INK))
for p, m in zip(bp['boxes'], MACHINES * 2):
    p.set_facecolor(MC[m]); p.set_alpha(.7); p.set_edgecolor(MC[m])
ax[2].axhline(8, color='#e34948', ls='--', lw=1); ax[2].text(7.4, 8.2, 'YOLO 최소 stride 8', color=INK2, fontsize=8, ha='right')
ax[2].set_xticks([2, 6], ['GT 박스 한 변', '결함 코어 한 변']); ax[2].set_ylabel('416 입력 기준 px')
ax[2].set_title('모델 입력(416px) 환산 크기')
save(fig, 'f4_size')

# ═══ F5. 위치: 제품 내부 분포 & 가장자리 ═══
fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
for m in MACHINES:
    o = obj[obj.machine == m]
    ax[0].scatter(o.cx, o.cy, s=6, color=MC[m], alpha=.5, lw=0, label=m)
ax[0].add_patch(Rectangle((.1, .1), .8, .8, fill=False, ls='--', ec='#e34948', lw=1))
ax[0].text(.12, .15, '이미지 가장자리 10% 경계', color=INK2, fontsize=8)
ax[0].set_xlim(0, 1); ax[0].set_ylim(1, 0); ax[0].set_aspect('equal')
ax[0].set_xlabel('cx'); ax[0].set_ylabel('cy'); ax[0].set_title('결함 위치 (정규화)'); ax[0].legend(markerscale=3, loc='lower right')
for m in MACHINES:
    ax[1].hist(obj[obj.machine == m].dist_prod_edge_px, bins=np.arange(30, 92, 2), histtype='step', lw=2, color=MC[m], label=m)
ax[1].set_xlabel('제품 외곽선까지 거리 (px)'); ax[1].set_ylabel('객체 수')
ax[1].set_title(f'제품 가장자리 거리: 최소 {obj.dist_prod_edge_px.min():.0f}px'); ax[1].legend()
ax[1].set_xlim(0, 92); ax[1].axvspan(0, 15, color='#e34948', alpha=.08); ax[1].text(1, ax[1].get_ylim()[1] * .9, '가장자리\n(샘플 0)', fontsize=8, color=INK2)
# 제품 평균 마스크 + 위치 (3호기)
s = img[(img.machine == '3호기') & (img.n_obj == 3)].stem.iloc[0]
gray, col, rgb = load_rgb(img_path(s, True)); clean = remove_markers(gray, col)
ax[2].imshow(clean, cmap='gray', vmin=0, vmax=255); ax[2].grid(False)
o = obj[obj.machine == '3호기']
ax[2].scatter(o.cx * o.W, o.cy * o.H, s=3, color='#eda100', alpha=.4, lw=0)
ax[2].set_title('3호기 전체 결함 위치 중첩 (배경: 샘플 1장)'); ax[2].axis('off')
save(fig, 'f5_position')

# ═══ F6. 대비·노이즈·배경(밀도) ═══
fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
for m in MACHINES:
    o = obj[obj.machine == m]
    ax[0].scatter(o.bg_wide, o.contrast, s=8, color=MC[m], alpha=.45, lw=0, label=m)
k = np.polyfit(obj.bg_wide, obj.contrast, 1)
xs = np.linspace(obj.bg_wide.min(), obj.bg_wide.max(), 10)
ax[0].plot(xs, np.polyval(k, xs), color=INK, lw=1.5, ls='--')
ax[0].text(.03, .95, f'r = {obj[["bg_wide","contrast"]].corr().iloc[0,1]:.2f}', transform=ax[0].transAxes, va='top', color=INK2)
ax[0].set_xlabel('결함 주변 배경 밝기 (낮을수록 두꺼움/고밀도)'); ax[0].set_ylabel('결함 대비 (gray level)')
ax[0].set_title('배경 투과량 ↔ 결함 대비'); ax[0].legend(markerscale=2, loc='lower right')
for m in MACHINES:
    ax[1].hist(obj[obj.machine == m].cnr, bins=np.arange(0, 38, 1.5), histtype='step', lw=2, color=MC[m], label=m)
ax[1].axvline(5, color='#e34948', ls='--', lw=1); ax[1].text(3.5, ax[1].get_ylim()[1] * .9, 'CNR 5\n(Rose 기준)', fontsize=8, color=INK2, ha='right')
ax[1].set_xlabel('CNR = 대비 / 로컬 노이즈 σ'); ax[1].set_ylabel('객체 수'); ax[1].set_title('검출 난이도 (CNR) 분포'); ax[1].legend()
t = obj[obj.n_obj == 3].groupby(['machine', 'strip']).cnr.median().unstack()
w = .25
for i, m in enumerate(MACHINES):
    ax[2].bar(np.arange(3) + (i - 1) * w, t.loc[m].values, width=w - .02, color=MC[m], label=m)
ax[2].set_xticks(range(3), ['띠 1 (상)', '띠 2 (중)', '띠 3 (하)'])
ax[2].set_ylabel('CNR 중앙값'); ax[2].set_title('띠(strip) 위치별 CNR — 3개 결함 제품'); ax[2].legend()
save(fig, 'f6_contrast')

# ═══ F7. 가장 어려운 / 쉬운 결함 패치 ═══
sel = pd.concat([obj.nsmallest(8, 'cnr'), obj.nlargest(8, 'cnr')])
fig, ax = plt.subplots(2, 8, figsize=(16, 4.6))
for k, (_, o) in enumerate(sel.iterrows()):
    gray, col, rgb = load_rgb(img_path(o.stem, o.labeled_in_400)); clean = remove_markers(gray, col)
    x, y, r = int(o.cx * o.W), int(o.cy * o.H), 12
    a = ax[k // 8, k % 8]
    a.imshow(clean[max(y - r, 0):y + r, max(x - r, 0):x + r], cmap='gray', vmin=40, vmax=160, interpolation='nearest')
    a.set_title(f'{o.machine} CNR {o.cnr:.1f}', fontsize=8); a.axis('off')
ax[0, 0].text(-4, 12, '하위 8', rotation=90, va='center', ha='right', fontsize=10, weight='bold')
ax[1, 0].text(-4, 12, '상위 8', rotation=90, va='center', ha='right', fontsize=10, weight='bold')
fig.suptitle('CNR 최하위 / 최상위 결함 (마커 제거, 24×24px, 동일 윈도우 40–160)', weight='bold')
save(fig, 'f7_patches')

# ═══ F8. 분할·누수 ═══
fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
sub_sizes = [15, 50, 100, 200, 300, 400]
test = img[img.split == 'test']
leak = [test.min_subset.le(s).sum() for s in sub_sizes]
ax[0].bar([str(s) for s in sub_sizes], leak, color='#e34948', width=.6, label='테스트 이미지 중 학습에 사용됨')
ax[0].bar([str(s) for s in sub_sizes], [60 - l for l in leak], bottom=leak, color='#d4d3ce', width=.6, label='미사용 (진짜 hold-out)')
for i, l in enumerate(leak): ax[0].text(i, l + 1, l, ha='center', color=INK2)
ax[0].set_xlabel('가중치 (last{N}.pt)'); ax[0].set_ylabel('테스트 60장 중'); ax[0].set_title('기존 가중치의 테스트셋 오염', pad=14); ax[0].legend(fontsize=8, loc='upper left')
ax[0].grid(axis='x', visible=False)

x = img.sort_values(['machine', 'ts']).copy()
x['gap'] = x.groupby('machine').ts.diff().dt.total_seconds()
x['session'] = (x.gap.isna() | (x.gap > 600)).cumsum()
ax[1].hist(np.log10(x.gap.dropna().clip(1)), bins=40, color='#8a8984')
ax[1].set_xticks([0, 1, 2, 3, 4, 5, 6], ['1s', '10s', '100s', '17m', '2.8h', '28h', '12d'])
ax[1].set_xlabel('같은 장비 연속 라벨 이미지 간 시간 간격'); ax[1].set_ylabel('쌍 수')
ax[1].set_title(f'세션 구조: {x.session.nunique()}개 세션, 세션당 ~6장 (4초 간격)')

Hb = np.array([[c == '1' for c in h] for h in img.dhash]); Ws = img.W.values
D = (Hb[:, None] != Hb[None]).sum(-1).astype(float); np.fill_diagonal(D, np.nan); D[Ws[:, None] != Ws[None]] = np.nan
sp = img.split.values
same_sess = x.set_index('stem').session.reindex(img.stem).values
S = same_sess[:, None] == same_sess[None]
ax[2].hist(D[S & ~np.isnan(D)], bins=np.arange(0, 80, 2), color='#2a78d6', alpha=.7, density=True, label='같은 세션 쌍')
ax[2].hist(D[~S & ~np.isnan(D)], bins=np.arange(0, 80, 2), color='#8a8984', alpha=.5, density=True, label='다른 세션 쌍 (같은 장비)')
ax[2].set_xlabel('dHash 해밍 거리 (256bit, 마커 제거본)'); ax[2].set_ylabel('밀도')
dS, dO = np.nanmedian(D[S & ~np.isnan(D)]), np.nanmedian(D[~S & ~np.isnan(D)])
ax[2].set_title(f'이미지 유사도: 세션 내 중앙값 {dS:.0f} vs 세션 간 {dO:.0f}'); ax[2].legend(fontsize=8)
print('dhash median same/other session', dS, dO)
save(fig, 'f8_leakage')

# ── 리포트용 수치 요약
x.to_csv(f'{OUT}/images_with_session.csv', index=False)
print('figures saved to', FIG)

# ═══ F9. 마커 색 분석: 장비별 구성, 3호기 시기별 변화, 겹친 박스 확대 ═══
mk2 = mk.merge(img[['stem', 'machine', 'date']], on='stem')
mk2['mix'] = np.where(mk2.color == 'red', '빨강 단색',
             np.where(mk2.color == 'blue', '파랑 단색',
             np.where(mk2.color.str.contains(r'\+'), '여러 색 겹침', '기타 단색')))
cats = ['빨강 단색', '파랑 단색', '기타 단색', '여러 색 겹침']
cc9 = {'빨강 단색': '#e34948', '파랑 단색': '#2a78d6', '기타 단색': '#eda100', '여러 색 겹침': '#4a3aa7'}
fig = plt.figure(figsize=(16, 4.4))
gs9 = fig.add_gridspec(1, 3, width_ratios=[1, 1.6, 1.1])
a = fig.add_subplot(gs9[0])
ct = pd.crosstab(mk2.machine, mk2.mix, normalize='index').reindex(columns=cats, fill_value=0)
left = np.zeros(3)
for c in cats:
    a.barh(MACHINES, ct.loc[MACHINES, c], left=left, color=cc9[c], height=.6, label=c, edgecolor=SURF, linewidth=2)
    left += ct.loc[MACHINES, c].values
a.invert_yaxis(); a.set_xlim(0, 1); a.set_xlabel('마커 비율'); a.set_title('장비별 마커 색 구성'); a.grid(axis='y', visible=False)
a.legend(loc='upper center', bbox_to_anchor=(.5, -.2), ncol=2, fontsize=8)

a = fig.add_subplot(gs9[1])
m3 = mk2[mk2.machine == '3호기'].copy()
m3['week'] = pd.to_datetime(m3.date.astype(str)).dt.to_period('W').dt.start_time
wk = pd.crosstab(m3.week, m3.mix, normalize='index').reindex(columns=cats, fill_value=0)
n_wk = m3.groupby('week').size()
bottom = np.zeros(len(wk))
for c in cats:
    a.bar(range(len(wk)), wk[c], bottom=bottom, color=cc9[c], width=.8, edgecolor=SURF, linewidth=1)
    bottom += wk[c].values
a.set_xticks(range(len(wk)), [f'{d:%m/%d}\n(n={n_wk[d]})' for d in wk.index], fontsize=8)
a.set_ylabel('마커 비율'); a.set_title('3호기 주별 마커 색 구성 (자홍색은 6월에만 등장)'); a.grid(axis='x', visible=False)

a = fig.add_subplot(gs9[2])
ex = mk2[(mk2.machine == '3호기') & (mk2.color == 'blue+magenta+red')].iloc[0]
_, _, rgb9 = load_rgb(img_path(ex.stem, True))
p = 6
a.imshow(rgb9[ex.y0 - p:ex.y0 + ex.h + p, ex.x0 - p:ex.x0 + ex.w + p].astype(np.uint8), interpolation='nearest')
a.set_xticks([]); a.set_yticks([]); a.grid(False)
a.set_title(f'겹친 마커 확대 ({ex.w}×{ex.h}px)\n18×18 박스 여러 개가 1~2px 어긋나 겹침', fontsize=10)
save(fig, 'f9_marker_colors')
print('f9 saved')
