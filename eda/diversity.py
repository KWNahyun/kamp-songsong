"""데이터 다양성 정량화: 결함의 크기·위치·형상·밀도(감쇠)·대비(배경)가 실제로 얼마나 다양한가
입력: eda/out/objects.csv (결함별 기본 통계), data/clean/images (마커 제거본)
결함별 추가 측정 (결함 코어 중심 12×12 창):
  감쇠율 a = 1 - I / I_bg      (I_bg = 결함 주변 국소 배경. 0 = 투과 그대로, 1 = 완전 차단)
  광학 밀도 OD = -ln(1 - a_peak)   (Beer-Lambert: OD = Σ μ·t, 이물 재질 밀도 × 두께에 비례)
  크기 = 반치(최대 감쇠의 절반 이상) 픽셀 수, 형상 = 평균 결함 패치와의 정규화 상관(NCC)
  위치 = 제품 영역 외접 사각형 기준 정규화 좌표 (u, v), 제품 가장자리까지 거리
제품 비교: 장비별 이미지 40장의 제품 픽셀 밝기 분포, 제품 가장자리 36px 안 면적 비율
출력: eda/diversity/defect_features.csv, product_summary.csv, eda/fig/f10_diversity.png, f11_defect_patches.png
"""
import os, sys
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

K = '/data/knhyun/KAMP'
sys.argv = ['x']
exec(open(f'{K}/eda/extract.py').read().split('# ─────────────────────────── 1.')[0])   # product_mask
OUT, FIG = f'{K}/eda/diversity', f'{K}/eda/fig'
os.makedirs(OUT, exist_ok=True)
R = 5
MC = ['1호기', '2호기', '3호기']
COL = {'1호기': '#2a78d6', '2호기': '#1baf7a', '3호기': '#eb6834'}

o = pd.read_csv(f'{K}/eda/out/objects.csv')
feats, patches, prod = [], {}, {m: dict(lv=[], edge=[], cover=[]) for m in MC}
n_prod_img = {m: 0 for m in MC}
for s, g in o.groupby('stem'):
    im8 = cv2.imread(f'{K}/data/clean/images/{s}.png', 0); im = im8.astype(np.float32)
    pm = product_mask(im8); ys, xs = np.where(pm)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    mc = g.machine.iloc[0]
    if n_prod_img[mc] < 40:   # 제품 쪽 통계는 장비별 40장
        n_prod_img[mc] += 1
        dt = cv2.distanceTransform(pm.astype(np.uint8), cv2.DIST_L2, 5)
        prod[mc]['lv'].append(im8[pm]); prod[mc]['edge'].append((dt[pm] < 36).mean())
        m = np.zeros_like(im8)
        for r in g.itertuples():
            cv2.circle(m, (int(r.cx * r.W), int(r.cy * r.H)), 10, 1, -1)
        prod[mc]['cover'].append((m.astype(bool) & pm).sum() / pm.sum())
    for r in g.itertuples():
        px = int(round(r.cx * r.W + r.dark_dx)); py = int(round(r.cy * r.H + r.dark_dy))
        row = dict(stem=s, obj=r.obj, u=(px - x0) / (x1 - x0), v=(py - y0) / (y1 - y0))
        P = im[py - R - 1:py + R + 1, px - R - 1:px + R + 1]
        if P.shape == (2 * R + 2, 2 * R + 2):
            att = 1 - P / r.bg_local
            w = np.clip(att, 0, None); w[w < 0.5 * att.max()] = 0
            row.update(att_peak=att.max(), od_peak=-np.log(max(1e-3, 1 - att.max())), fwhm_px=int((w > 0).sum()))
            patches[(s, r.obj)] = att
        feats.append(row)
d = o.merge(pd.DataFrame(feats), on=['stem', 'obj'])
for mc, g in d.groupby('machine'):
    keys = [(r.stem, r.obj) for r in g.itertuples() if (r.stem, r.obj) in patches]
    A = np.stack([patches[k] / max(patches[k].max(), 1e-6) for k in keys])
    mean = A.mean(0); mz = (mean - mean.mean()) / mean.std()
    ncc = pd.Series([((a - a.mean()) / a.std() * mz).mean() for a in A], index=pd.MultiIndex.from_tuples(keys))
    idx = pd.MultiIndex.from_frame(g[['stem', 'obj']])
    d.loc[g.index, 'ncc_mean'] = ncc.reindex(idx).values
    np.save(f'{OUT}/mean_patch_{mc}.npy', mean)
d.to_csv(f'{OUT}/defect_features.csv', index=False)
ps = pd.DataFrame([dict(machine=m, prod_p5=np.percentile(np.concatenate(prod[m]['lv']), 5),
                        prod_p50=np.percentile(np.concatenate(prod[m]['lv']), 50),
                        prod_p95=np.percentile(np.concatenate(prod[m]['lv']), 95),
                        prod_edge36_frac=np.mean(prod[m]['edge']), defect_site_cover=np.mean(prod[m]['cover'])) for m in MC])
ps.to_csv(f'{OUT}/product_summary.csv', index=False)

# ── 요약 표
q = lambda x: f'{x.quantile(.05):.2f} ~ {x.quantile(.95):.2f} (중앙 {x.median():.2f})'
summ = pd.DataFrame({m: {
    '크기: 반치 면적 px': q(g.fwhm_px), '크기: 코어 한 변 (416 입력 기준, px)': q(np.sqrt(g.fwhm_px) * 416 / g.W),
    '위치: u (제품 가로)': q(g.u), '위치: v (제품 세로)': q(g.v), '위치: 제품 가장자리 거리 px': q(g.dist_prod_edge_px),
    '형상: 평균 패치와 NCC': q(g.ncc_mean.dropna()), '밀도: 최대 감쇠율': q(g.att_peak), '밀도: 광학 밀도 OD': q(g.od_peak),
    '대비: 국소 배경 밝기': q(g.bg_local), '대비: 밝기 차': q(g.contrast), '대비: CNR': q(g.cnr)}
    for m, g in d.groupby('machine')})
summ.to_csv(f'{OUT}/summary.csv')
pd.set_option('display.width', 250); print(summ.to_string()); print(ps.round(3).to_string())

# ── 그림 f10: 조건별 분포
plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False, 'axes.spines.top': False,
                     'axes.spines.right': False, 'font.size': 10, 'axes.titleweight': 'bold', 'axes.titlesize': 11})
fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))
a = ax[0, 0]
for m in MC:
    g = d[d.machine == m]; a.scatter(g.u, g.v, s=10, alpha=.5, color=COL[m], label=m)
a.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, ec='#52514e', ls='--')); a.set_xlim(-.05, 1.05); a.set_ylim(1.05, -.05)
a.set_aspect('equal'); a.set_title('위치: 제품 영역 안의 결함 위치 (제품 외접 사각형 = 점선)'); a.set_xlabel('u'); a.set_ylabel('v'); a.legend(fontsize=9)
a = ax[0, 1]
for m in MC:
    a.hist(d[d.machine == m].fwhm_px, bins=np.arange(.5, 16.5, 1), alpha=.55, color=COL[m], label=m)
a.set_title('크기: 결함 반치 면적 (px)'); a.set_xlabel('px'); a.legend(fontsize=9)
a = ax[0, 2]
for m in MC:
    a.hist(d[d.machine == m].att_peak, bins=30, alpha=.55, color=COL[m], label=m)
a.set_title('밀도: 결함 중심의 감쇠율 (1 - I/I_배경)'); a.set_xlim(0, 1); a.set_xlabel('감쇠율'); a.legend(fontsize=9)
a = ax[1, 0]
for i, m in enumerate(MC):
    lv = np.concatenate(prod[m]['lv']); g = d[d.machine == m]
    a.violinplot([lv], positions=[i * 2], widths=.9, showextrema=False)
    a.scatter(np.full(len(g), i * 2 + .6) + np.random.default_rng(0).normal(0, .06, len(g)), g.bg_local, s=4, color=COL[m], alpha=.4)
a.set_xticks([0, 2, 4], [f'{m}\n제품 전체 | 결함 위치' for m in MC]); a.set_ylabel('밝기 (0~255)')
a.set_title('대비(배경): 결함이 놓인 배경 밝기 vs 제품 전체 밝기')
a = ax[1, 1]
for m in MC:
    a.hist(d[d.machine == m].cnr, bins=30, alpha=.55, color=COL[m], label=m)
a.axvline(5, color='#e34948', ls='--', lw=1); a.text(5.3, a.get_ylim()[1] * .9, 'CNR 5 미만 = 0개', color='#e34948', fontsize=9)
a.set_title('대비: CNR (밝기 차 / 잡음)'); a.set_xlabel('CNR'); a.legend(fontsize=9)
a = ax[1, 2]
for m in MC:
    a.hist(d[d.machine == m].dist_prod_edge_px, bins=30, alpha=.55, color=COL[m], label=m)
a.axvline(36, color='#e34948', ls='--', lw=1)
a.text(37, a.get_ylim()[1] * .9, f'제품 면적의 {ps.prod_edge36_frac.min():.0%}~{ps.prod_edge36_frac.max():.0%}가\n가장자리 36px 안 → 결함 0개', color='#e34948', fontsize=9)
a.set_title('위치: 제품 가장자리까지 거리 (px)'); a.set_xlabel('px'); a.legend(fontsize=9)
fig.suptitle('데이터 다양성: GT 결함 1,147개의 크기·위치·밀도·대비 분포', weight='bold', y=1.0)
fig.tight_layout(); fig.savefig(f'{FIG}/f10_diversity.png', dpi=120, bbox_inches='tight'); plt.close(fig)

# ── 그림 f11: 형상 — 장비별 평균 패치 + 무작위 결함 12개 (감쇠율 맵, 확대)
rng = np.random.default_rng(0)
fig, ax = plt.subplots(3, 13, figsize=(17, 4.6))
for i, m in enumerate(MC):
    mean = np.load(f'{OUT}/mean_patch_{m}.npy')
    ax[i, 0].imshow(mean, cmap='gray_r', vmin=0, vmax=1, interpolation='nearest'); ax[i, 0].set_title(f'{m} 평균', fontsize=9)
    keys = [(r.stem, r.obj) for r in d[d.machine == m].itertuples() if (r.stem, r.obj) in patches]
    for j, k in enumerate(rng.choice(len(keys), 12, replace=False)):
        ax[i, j + 1].imshow(patches[keys[k]], cmap='gray_r', vmin=0, vmax=.7, interpolation='nearest')
    for a in ax[i]:
        a.set_xticks([]); a.set_yticks([]); [sp.set_visible(False) for sp in a.spines.values()]
fig.suptitle('형상: 결함 12×12px 감쇠율 맵 (진할수록 많이 가림). 모든 결함이 1~4px 점', weight='bold')
fig.savefig(f'{FIG}/f11_defect_patches.png', dpi=120, bbox_inches='tight'); plt.close(fig)
print('saved f10, f11')
