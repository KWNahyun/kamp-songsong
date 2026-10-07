"""v2 실험 결과 그림 → eval/fig_v2/*.png
x1: ① 마커 처리 (마커 없는 입력 성능 + 흔적 검사)
x2: ② GT 없는 데이터 (베이스라인 / 학습량 대조군 / 의사 라벨 학생)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

K = '/data/knhyun/KAMP'
FIG = f'{K}/eval/fig_v2'; os.makedirs(FIG, exist_ok=True)
SURF, INK2, GRID = '#fcfcfb', '#52514e', '#e4e3df'
plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False, 'figure.facecolor': SURF,
                     'axes.facecolor': SURF, 'savefig.facecolor': SURF, 'axes.grid': True, 'grid.color': GRID,
                     'axes.axisbelow': True, 'axes.spines.top': False, 'axes.spines.right': False,
                     'legend.frameon': False, 'axes.titleweight': 'bold', 'axes.titlesize': 12,
                     'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'font.size': 10})
m = pd.read_csv(f'{K}/eval/out_v2/metrics.csv')
m['cond'] = m.run.str.replace(r'_s\d$', '', regex=True)


def bars(a, conds, labels, col, split='test', inp='clean', colors=None, title='', ylim=(0, 1.05)):
    x = np.arange(len(conds))
    mu = [m[(m.cond == c) & (m.split == split) & (m.input == inp)][col].mean() for c in conds]
    sd = [m[(m.cond == c) & (m.split == split) & (m.input == inp)][col].std() for c in conds]
    b = a.bar(x, mu, yerr=sd, capsize=4, color=colors, width=.6, error_kw=dict(ecolor=INK2, lw=1))
    off = (ylim[1] - ylim[0]) * 0.015
    for bb, v, e in zip(b, mu, sd):
        a.text(bb.get_x() + bb.get_width() / 2, v + (e if e == e else 0) + off, f'{v:.3f}', ha='center', va='bottom', fontsize=9, color=INK2)
    a.set_xticks(x, labels); a.set_ylim(*ylim); a.set_title(title); a.grid(axis='x', visible=False)


# x1
C = ['v2/M_marked', 'v2/M_masked', 'v2/B']; L = ['원본\n(마커 포함)', '마스킹\n(회색 칠)', '인페인팅']
col = ['#8a8984', '#eda100', '#2a78d6']
fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
bars(ax[0], C, L, 'AP50', colors=col, title='마커 없는 입력의 test AP@0.5')
bars(ax[1], C, L, 'R_at25', colors=col, title='마커 없는 입력의 test 재현율 (conf 0.25)')
tr = pd.read_csv(f'{K}/eval/marker_v2/trace_sites_mask.csv'); ti = pd.read_csv(f'{K}/eval/marker_v2/trace_sites_inpaint.csv')
for d in (tr, ti): d['cond'] = d.run.str.replace(r'_s\d$', '', regex=True)
w = .38
for i, (d, lab, c) in enumerate([(tr, '회색 링 (마스킹 흔적)', '#e34948'), (ti, '인페인팅 링', '#1baf7a')]):
    v = [d[d.cond == k].groupby('run').apply(lambda g: (g.conf_traced >= .1).mean()) for k in C]
    b = ax[2].bar(np.arange(3) + (i - .5) * w, [x.mean() for x in v], yerr=[x.std() for x in v], width=w - .03, color=c, label=lab, capsize=3, error_kw=dict(ecolor=INK2, lw=1))
    for bb, x in zip(b, v): ax[2].text(bb.get_x() + bb.get_width() / 2, x.mean() + x.std() + .012, f'{x.mean():.0%}', ha='center', va='bottom', fontsize=9, color=INK2)
ax[2].set_xticks(range(3), L); ax[2].set_ylim(0, 0.7); ax[2].set_title('결함 없는 곳에 흔적만 넣었을 때 반응 (conf ≥ 0.1)')
ax[2].legend(fontsize=9); ax[2].grid(axis='x', visible=False)
fig.suptitle('① 마커 처리 방식 비교 (seed 3개 평균 ± 표준편차)', weight='bold', y=1.03)
fig.savefig(f'{FIG}/x1_marker.png', dpi=130, bbox_inches='tight'); plt.close(fig)

# x2
C = ['v2/B', 'v2/U0ctrl', 'v2/U1_p990', 'v2/U1_p1000']
L = ['베이스라인\n(GT 350장)', '학습량 대조군\n(GT 반복)', '의사 라벨\n(정밀도 0.99 기준)', '의사 라벨\n(정밀도 1.00 기준)']
col = ['#2a78d6', '#8a8984', '#eb6834', '#eda100']
fig, ax = plt.subplots(1, 2, figsize=(14, 4.3))
bars(ax[0], C, L, 'AP50', colors=col, title='test AP@0.5', ylim=(0.85, 0.97))
bars(ax[1], C, L, 'AP50', split='val', colors=col, title='val AP@0.5', ylim=(0.85, 0.99))
fig.suptitle('② GT 없는 데이터의 의사 라벨 활용 (seed 3개 평균 ± 표준편차)', weight='bold', y=1.03)
fig.savefig(f'{FIG}/x2_unlabeled.png', dpi=130, bbox_inches='tight'); plt.close(fig)

# x3: 의사 라벨 방법 비교 (train 3-fold 교차 검증, 공식 TXT 기준)
p = pd.read_csv(f'{K}/eval/pseudo_v2/pl_methods_cv.csv')
g = p.groupby('method').sum(numeric_only=True).reindex(['MK', 'H2', 'H1', 'T0', 'T1', 'T2', 'T3', 'T4'])
L = ['마커\n(MK)', '혼합\n(H2)', '혼합\n(H1)', '교사\nT0', '교사\nT1', '교사\nT2', '교사\nT3\n(현재)', '교사\nT4']
col = ['#8a8984'] + ['#eda100'] * 2 + ['#2a78d6'] * 5
panels = [('박스 정밀도 (중심거리 ≤ 8px)', g.tp_ctr / g.pl_boxes, (0.9, 1.005)),
          ('채택 이미지의 완전 일치 비율', g.n_exact / g.kept, (0.75, 1.02)),
          ('전체 GT 결함 중 의사 라벨로 얻은 비율', g.hit_kept / g.gt_all, (0.9, 1.005))]
fig, ax = plt.subplots(1, 3, figsize=(17, 4.3))
for a, (t, v, yl) in zip(ax, panels):
    b = a.bar(range(len(v)), v.values, color=col, width=.6)
    for bb, x in zip(b, v.values): a.text(bb.get_x() + bb.get_width() / 2, x + (yl[1] - yl[0]) * .01, f'{x:.3f}', ha='center', va='bottom', fontsize=8.5, color=INK2)
    a.set_xticks(range(len(v)), L, fontsize=8.5); a.set_ylim(*yl); a.set_title(t); a.grid(axis='x', visible=False)
fig.suptitle('② 의사 라벨 방법 비교 — train 350장 3-fold 교차 검증, 공식 TXT와 비교 (세 조각 합산)', weight='bold', y=1.03)
fig.savefig(f'{FIG}/x3_pl_methods.png', dpi=130, bbox_inches='tight'); plt.close(fig)

# x4: 마커 제거 방법 교차표 (학습 방법 × test 입력) + 흔적 검사
TR = [('v2/M_marked', '원본'), ('v2/M_masked', '마스킹'), ('v2/B', 'Telea (기준)'), ('v2/M_rm_ns', 'Navier-Stokes'),
      ('v2/M_rm_dil', '2px 확장+Telea'), ('v2/M_rm_bgfill', '배경 채우기'), ('v2/M_rm_biharm', 'Biharmonic')]
TE = [('marked', '원본'), ('masked', '마스킹'), ('clean', 'Telea'), ('rm_ns', 'NS'), ('rm_dil', '확장+Telea'),
      ('rm_bgfill', '배경 채우기'), ('rm_biharm', 'Biharmonic')]
if all((m.cond == c).any() for c, _ in TR):
    fig, ax = plt.subplots(1, 3, figsize=(21, 5.4), gridspec_kw=dict(width_ratios=[1, 1, 1], wspace=0.28))
    for a, col_, t in [(ax[0], 'AP50', 'test AP@0.5'), (ax[1], 'R_at25', 'test 재현율 (conf 0.25)')]:
        M = np.array([[m[(m.cond == c) & (m.split == 'test') & (m.input == i)][col_].mean() for i, _ in TE] for c, _ in TR])
        a.imshow(M, cmap='Blues', vmin=0, vmax=1, aspect='auto'); a.grid(False)
        for r in range(M.shape[0]):
            for c in range(M.shape[1]):
                a.text(c, r, f'{M[r, c]:.2f}', ha='center', va='center', fontsize=8.5, color='white' if M[r, c] > .6 else '#1c1b19',
                       weight='bold' if TR[r][0].split('/')[-1] in ('B',) and False else 'normal')
        for k in range(7): a.add_patch(plt.Rectangle((k - .45, k - .45), .9, .9, fill=False, ec='#eda100', lw=2))
        a.set_xticks(range(len(TE)), [x for _, x in TE], rotation=30, ha='right'); a.set_yticks(range(len(TR)), [x for _, x in TR])
        a.set_xlabel('test 입력 (마커 처리)'); a.set_ylabel('학습 입력 (마커 처리)' if a is ax[0] else ''); a.set_title(t + '  (노란 테두리 = 학습·평가 같은 방식)')
    # 흔적 검사: 각 모델에 자기 제거 방법의 링 / 회색 링
    own = {'v2/M_marked': 'inpaint', 'v2/M_masked': 'mask', 'v2/B': 'inpaint', 'v2/M_rm_ns': 'rm_ns', 'v2/M_rm_dil': 'rm_dil',
           'v2/M_rm_bgfill': 'rm_bgfill', 'v2/M_rm_biharm': 'rm_biharm'}
    tr_ = {md: pd.read_csv(f'{K}/eval/marker_v2/trace_sites_{md}.csv') for md in set(own.values()) | {'mask'}}
    for d in tr_.values(): d['cond'] = d.run.str.replace(r'_s\d$', '', regex=True)
    rate = lambda d, c: d[d.cond == c].groupby('run').apply(lambda g: (g.conf_traced >= .1).mean())
    w = .38
    for i, (lab, cc, f) in enumerate([('자기 방식으로 지운 링 (원본 모델은 Telea 링)', '#1baf7a', lambda c: tr_[own[c]]), ('회색 링', '#e34948', lambda c: tr_['mask'])]):
        v = [rate(f(c), c) for c, _ in TR]
        b = ax[2].bar(np.arange(7) + (i - .5) * w, [x.mean() for x in v], yerr=[x.std() for x in v], width=w - .03, color=cc, label=lab, capsize=3, error_kw=dict(ecolor=INK2, lw=1))
        for bb, x in zip(b, v): ax[2].text(bb.get_x() + bb.get_width() / 2, x.mean() + x.std() + .01, f'{x.mean():.0%}', ha='center', va='bottom', fontsize=8.5, color=INK2)
    ax[2].set_xticks(range(7), [x for _, x in TR], rotation=30, ha='right'); ax[2].set_ylim(0, .6); ax[2].legend(fontsize=9)
    ax[2].set_title('결함 없는 곳에 흔적만 넣었을 때 반응 (conf ≥ 0.1)'); ax[2].grid(axis='x', visible=False)
    fig.suptitle('① 마커 제거 방법 확장 — 학습 방식 × 평가 방식 교차표 (seed 3개 평균)', weight='bold', y=1.02)
    fig.savefig(f'{FIG}/x4_marker_methods.png', dpi=130, bbox_inches='tight'); plt.close(fig)
print('saved', sorted(os.listdir(FIG)))
