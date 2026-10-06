"""합성 test 데이터 시각화 (보고서 §4.3·§4.5·§4.6) → eda/fig/s3_synth_replica.png, s4_synth_position.png, s5_synth_gvxr.png
입력: data/clean, data/synth_val (복제 검증 세트), handoff/kamp_synth_test_v1 (prep/synth/build_synth_test.py)
모든 확대 영상은 최근접 보간, 같은 그림의 같은 행은 같은 회색조 범위로 표시 (대비를 부풀리지 않음)
"""
import os
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

K = '/data/knhyun/KAMP'; FIG = f'{K}/eda/fig'; PKG = f'{K}/handoff/kamp_synth_test_v1'
SURF, INK, INK2, MUTED = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8984'
C_CANON, C_RAND = '#eb6834', '#2a78d6'
plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False, 'figure.facecolor': SURF,
                     'savefig.facecolor': SURF, 'text.color': INK, 'font.size': 9, 'axes.titlesize': 9})
split = pd.read_csv(f'{K}/data/splits/split.csv').set_index('stem')
P = pd.read_csv(f'{K}/eval/synth/replica_params.csv')
I = pd.read_csv(f'{PKG}/images.csv'); O = pd.read_csv(f'{PKG}/objects.csv')


def crop(img, cx, cy, r):
    p = np.pad(img, r + 2, mode='edge'); x, y = int(round(cx)) + r + 2, int(round(cy)) + r + 2
    return p[y - r:y + r, x - r:x + r]


def show(ax, c, lo, hi, title=None, color=INK):
    ax.imshow(c, cmap='gray', vmin=lo, vmax=hi, interpolation='nearest')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(MUTED); s.set_linewidth(0.6)
    if title:
        ax.set_title(title, color=color, fontsize=8, pad=3)


def save(fig, name):
    fig.savefig(f'{FIG}/{name}.png', bbox_inches='tight', dpi=150); plt.close(fig); print(f'{FIG}/{name}.png')


# ── s3: 합성 타당성 검증 — 같은 자리의 실제 결함과 세 가지 합성 (test, 장비별 1개, 대비 중앙값에 가까운 결함)
cols = [('clean', '실제 결함'), ('synth_val/erased', '결함 지움'), ('synth_val/T_cross', '옮겨 심기'),
        ('synth_val/R1_cal', '매개변수 합성'), ('synth_val/G_fit', 'gVXR 구')]
fig, ax = plt.subplots(3, len(cols), figsize=(len(cols) * 1.55, 3 * 1.75))
for i, m in enumerate(['1호기', '2호기', '3호기']):
    p = P[(P.split == 'test') & (P.machine == m)]
    r = p.iloc[(p.contrast - p.contrast.median()).abs().argsort().iloc[0]]
    row = []
    for d, _ in cols:
        row.append(crop(cv2.imread(f'{K}/data/{d}/images/{r.stem}.png', 0).astype(float), r.px + 1, r.py + 1, 12))
    lo, hi = min(c.min() for c in row), max(c.max() for c in row)
    for j, (c, (_, t)) in enumerate(zip(row, cols)):
        show(ax[i, j], c, lo, hi, t if i == 0 else None)
    ax[i, 0].set_ylabel(f'{m}\n대비 {r.contrast:.0f}', fontsize=8, color=INK2)
fig.suptitle('합성 타당성 검증: 실제 결함을 지우고 같은 자리에 합성 결함을 다시 넣음 (24×24px)', fontsize=10, weight='bold', x=0.02, ha='left')
fig.tight_layout(); save(fig, 's3_synth_replica')

# ── s4: 위치 축 — 원래 자리(띠 끝) vs 제품 안 무작위 자리, 농도 k = 1 / 0.5
stem = I[(I.split == 'test') & (I.machine == 'M2') & (I.set == 'pos_random_k1.0')].source_image.iloc[0]
oc = O[O.image_id == f'{stem}__pos_canon_k1.0']; orr = O[O.image_id == f'{stem}__pos_random_k1.0__p0']
full = cv2.imread(f'{PKG}/images/test/{stem}__pos_random_k1.0__p0.png', 0)
fig = plt.figure(figsize=(9.6, 4.2)); gs = fig.add_gridspec(2, 5, width_ratios=[2.6, 1, 1, 1, 0.05], wspace=0.22, hspace=0.25)
a = fig.add_subplot(gs[:, 0]); a.imshow(full, cmap='gray', vmin=0, vmax=255); a.set_xticks([]); a.set_yticks([])
for o in oc.itertuples():
    a.add_patch(Circle((o.obj_cx, o.obj_cy), 7, fill=False, ec=C_CANON, lw=1.3))
for o in orr.itertuples():
    a.add_patch(Rectangle((o.obj_cx - 7, o.obj_cy - 7), 14, 14, fill=False, ec=C_RAND, lw=1.3))
a.set_title(f'test 영상 1장 ({stem}, 2호기)\n주황 원 = 원래 결함 자리(띠 끝), 파랑 네모 = 무작위 자리 (회차 0)', fontsize=8, color=INK2)
bg = cv2.imread(f'{PKG}/images/test/{stem}__erased.png', 0).astype(float)
for i, (kind, o, col, lab) in enumerate([('canon', oc.iloc[0], C_CANON, '원래 자리'), ('random', orr.iloc[0], C_RAND, '무작위 자리')]):
    cs = [crop(bg, o.obj_cx, o.obj_cy, 10)]
    for k in [1.0, 0.5]:
        iid = f'{stem}__pos_{kind}_k{k}' + ('__p0' if kind == 'random' else '')
        cs.append(crop(cv2.imread(f'{PKG}/images/test/{iid}.png', 0).astype(float), o.obj_cx, o.obj_cy, 10))
    lo, hi = min(c.min() for c in cs), max(c.max() for c in cs)
    for j, (c, t) in enumerate(zip(cs, ['결함 없음 (지운 배경)', '옮겨 심기 k = 1', '옮겨 심기 k = 0.5'])):
        ax_ = fig.add_subplot(gs[i, j + 1]); show(ax_, c, lo, hi, t if i == 0 else None)
        if j == 0:
            ax_.set_ylabel(lab, color=col, fontsize=9, weight='bold')
fig.suptitle('위치 축 스트레스: 같은 실제 결함 맵을 띠 끝이 아닌 자리에 옮겨 심음 (확대 20×20px)', fontsize=10, weight='bold', x=0.02, ha='left', y=1.06)
save(fig, 's4_synth_position')

# ── s5: gVXR 크기·형상·재질 — 같은 자리(test 1장, 회차 0, 첫 자리)에 조건만 바꿔 넣음
g = I[(I.split == 'test') & (I.machine == 'M1') & (I.set == 'gvxr_size_x1')].source_image.iloc[3]
rows = [('크기 (SUS304 구, 시편 대비 배수)', [f'gvxr_size_x{k}' for k in ('0.5', '0.75', '1', '1.5', '2', '3', '5')], ['× 0.5', '× 0.75', '× 1 (시편)', '× 1.5', '× 2', '× 3', '× 5']),
        ('형상 (SUS304, 부피 = 시편)', [f'gvxr_shape_{s}' for s in ('sphere', 'cube', 'irregular', 'plate', 'wire10', 'wire20')], ['구', '정육면체', '불규칙 조각', '판', '선 (10배)', '선 (20배)'])] + \
       [(f'재질: {n} 구', [f'gvxr_material_{m}_x{k}' for k in (1, 2, 4, 8)], [f'× {k}' for k in (1, 2, 4, 8)])
        for m, n in [('SUS304', 'SUS304'), ('Al', '알루미늄'), ('glass', '유리'), ('stone', '돌'), ('bone', '뼈'), ('plastic', '플라스틱')]]
o0 = O[O.image_id == f'{g}__gvxr_size_x1__p0'].iloc[0]
bgc = crop(cv2.imread(f'{PKG}/images/test/{g}__erased.png', 0).astype(float), o0.obj_cx, o0.obj_cy, 16)
lo, hi = 0, bgc.max()
ncol = 8
fig, ax = plt.subplots(len(rows), ncol, figsize=(ncol * 1.15, len(rows) * 1.42))
for i, (lab, sets, titles) in enumerate(rows):
    show(ax[i, 0], bgc, lo, hi, '이물 없음' if i == 0 else None)
    ax[i, 0].set_ylabel(lab.replace(' (', '\n('), fontsize=7.5, color=INK2, rotation=0, ha='right', va='center', labelpad=4)
    for j in range(1, ncol):
        if j - 1 >= len(sets):
            ax[i, j].axis('off'); continue
        iid = f'{g}__{sets[j - 1]}__p0'; o = O[O.image_id == iid].iloc[0]
        show(ax[i, j], crop(cv2.imread(f'{PKG}/images/test/{iid}.png', 0).astype(float), o0.obj_cx, o0.obj_cy, 16), lo, hi)
        ax[i, j].set_title(f'{titles[j - 1]}\n대비 {o.contrast:.0f}', fontsize=7, pad=2, color=INK if o.contrast >= 10 else MUTED)
fig.suptitle(f'gVXR 크기·형상·재질 축: 같은 자리에 조건만 바꿔 넣음 (test {g}, 1호기, 32×32px, 회색조 0~{hi:.0f} 고정)\n'
             '실제 결함의 대비는 17~59. 대비 10 미만(회색 글씨)은 잡음·배경 구조 수준', fontsize=9, weight='bold', x=0.02, ha='left')
fig.tight_layout(); save(fig, 's5_synth_gvxr')
