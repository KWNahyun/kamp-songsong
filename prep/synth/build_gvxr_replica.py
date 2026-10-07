"""gVXR 합성 타당성 검증용 복제 데이터 (build_replica.py 의 gVXR 판) — 실제 결함을 지운 배경의 같은 자리에 gVXR 기준 이물을 넣는다
기준 이물 = SUS304 구 (gvxr_calibrate.py 가정), 장비별 픽셀 크기 p·구 지름 D·흐림 σ 는 eval/synth/gvxr/calibration.csv (train 결함으로 정함)
버전 (data/synth_val/<버전>/images/<stem>.png):
  G_gen  : 장비별 보정값만으로 (결함별 조정 없음, 위치만 결함별 맞춤 중심 x0, y0) → 생성 모델 검증 (R2_mult 에 해당)
  G_fit  : 결함마다 구 지름을 바꿔 그 결함의 적분 광학 밀도 M 에 맞춤 (p·σ 는 장비별) → 형상·흐림 모형 검증 (R1 에 해당)
  H_gvxr : 공정 비교. H_real 과 같은 결함 픽셀 영역(맞춤 상자 > 최대의 10%)만 G_fit 으로, 나머지는 지운 배경
배경: data/synth_val/erased (build_replica.py 가 만든 지운 배경을 8bit 로 저장한 것). 다른 버전은 반올림 전 실수 배경을 썼으므로 반올림이 한 번 더 들어감(±0.5 회색조)
"""
import os, sys
import numpy as np
import pandas as pd
import cv2
sys.path.insert(0, '/data/knhyun/KAMP/prep/synth')
import gvxr_core as G
from synth_core import box, insert

K = G.K; OUT = f'{K}/data/synth_val'
C = pd.read_csv(f'{K}/eval/synth/gvxr/calibration.csv').set_index('machine')
P = pd.read_csv(f'{K}/eval/synth/replica_params.csv')
N = 12
sim = G.Sim(os_=8, n_px=N)
# 장비별 M(D) 표: 구 지름 → 적분 광학 밀도 (부분 픽셀 위치 4곳 평균)
DG = np.round(np.arange(0.6, 5.01, 0.05), 3)
MD = {}
for m, c in C.iterrows():
    MD[m] = np.array([np.mean([sim.od_map(sim.path_length('sphere', d, off=o), c.material, c.p_mm, c.sigma).sum()
                               for o in [(0, 0), (0.5, 0), (0, 0.5), (0.5, 0.5)]]) for d in DG])
P['D_fit'] = [float(np.interp(r.M, MD[r.machine], DG)) for r in P.itertuples()]
P[['stem', 'obj', 'machine', 'M', 'D_fit']].to_csv(f'{K}/eval/synth/gvxr/replica_D_fit.csv', index=False)
print(P.groupby('machine').D_fit.describe().round(3))
for v in ['G_gen', 'G_fit', 'H_gvxr']:
    os.makedirs(f'{OUT}/{v}/images', exist_ok=True)


def place(G_, od, x0, y0):
    """od (N×N, 물체 중심 = 패치 좌표 N/2 - 0.5 + off) 를 영상의 (x0, y0) 에 더함"""
    H, W = G_.shape; X0, Y0 = int(round(x0)) - N // 2, int(round(y0)) - N // 2
    ys, xs = slice(max(Y0, 0), min(Y0 + N, H)), slice(max(X0, 0), min(X0 + N, W))
    G_[ys, xs] += od[ys.start - Y0:ys.stop - Y0, xs.start - X0:xs.stop - X0]


for s, p in P.groupby('stem'):
    bg = cv2.imread(f'{OUT}/erased/images/{s}.png', 0).astype(np.float32); H, W = bg.shape
    Gg, Gf = np.zeros((H, W), np.float32), np.zeros((H, W), np.float32); foot = np.zeros((H, W), bool)
    for r in p.itertuples():
        c = C.loc[r.machine]
        off = (r.x0 - round(r.x0) + 0.5, r.y0 - round(r.y0) + 0.5)
        place(Gg, sim.od_map(sim.path_length('sphere', c.D_px, off=off), c.material, c.p_mm, c.sigma), r.x0, r.y0)
        place(Gf, sim.od_map(sim.path_length('sphere', r.D_fit, off=off), c.material, c.p_mm, c.sigma), r.x0, r.y0)
        b1 = box((H, W), r.x0, r.y0, r.w, r.sigma); foot |= b1 > 0.1 * b1.max()
    cv2.imwrite(f'{OUT}/G_gen/images/{s}.png', insert(bg, Gg, OD=1.0))
    gf = insert(bg, Gf, OD=1.0); cv2.imwrite(f'{OUT}/G_fit/images/{s}.png', gf)
    hg = bg.astype(np.uint8).copy(); hg[foot] = gf[foot]; cv2.imwrite(f'{OUT}/H_gvxr/images/{s}.png', hg)
print('done', P.stem.nunique())
