"""gVXR 합성기 보정: 실제 결함(시편)을 '기준 이물'로 놓고 장비별 픽셀 크기 p(mm/px)와 기준 구 지름 D(px)를 정한다 — train 결함만 사용
가정: 실제 결함 = SUS304 구 (식품 X선 검사기 표준 시편에 흔한 재질). 장비 = 70 kVp, Al 2 mm (gvxr_core.py)
맞추는 값 (장비별 train 결함의 중앙값, replica_params.csv 의 상자 모형 맞춤 값 — 잡음 없는 값):
  적분 광학 밀도 M = Σ OD (px²),  최대 광학 밀도 OD_peak,  검출기 흐림 σ = 결함별 맞춤 σ 의 중앙값
방법: 구 지름 D(0.6~4 px) × 픽셀 크기 p(로그 격자) 에서 부분 픽셀 위치 4곳 평균의 (M, OD_peak) 를 계산해 상대 오차 제곱합이 최소인 (D, p)
민감도: 관전압 50·90 kVp, 기준 재질을 Fe / 유리(glass) 로 가정했을 때의 p — 가정에 따라 물리 단위(mm)가 얼마나 바뀌는지
출력: eval/synth/gvxr/calibration.csv (장비별 p, D, d_mm, 맞춤 오차), sensitivity.csv
"""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, '/data/knhyun/KAMP/prep/synth')
import gvxr_core as G

K = G.K; OUT = f'{K}/eval/synth/gvxr'; os.makedirs(OUT, exist_ok=True)
split = pd.read_csv(f'{K}/data/splits/split.csv').set_index('stem')
P = pd.read_csv(f'{K}/eval/synth/replica_params.csv'); P = P[P.stem.map(split.split) == 'train']
tgt = P.groupby('machine').agg(M=('M', 'median'), OD_peak=('OD_peak', 'median'), sigma=('sigma', 'median'), w=('w', 'median'), n=('M', 'size'))
print(tgt.round(3))
D_GRID = np.round(np.arange(0.6, 4.01, 0.1), 3)
P_GRID = np.logspace(-2.5, 0.5, 90)
OFFS = [(a, b) for a in (0, 0.5) for b in (0, 0.5)]


def stats_grid(sim, material, sigma):
    """(D, p) 격자의 평균 (M, OD_peak)"""
    out = np.zeros((len(D_GRID), len(P_GRID), 2))
    for i, d in enumerate(D_GRID):
        Ls = [sim.path_length('sphere', d, off=o) for o in OFFS]
        ods = np.stack([sim.od_maps_p(L, material, P_GRID, sigma) for L in Ls])   # (오프셋, p, n, n)
        out[i, :, 0] = ods.sum((2, 3)).mean(0); out[i, :, 1] = ods.max((2, 3)).mean(0)
    return out


def fit(sim, material, t):
    S = stats_grid(sim, material, t.sigma)
    err = ((S[..., 0] - t.M) / t.M) ** 2 + ((S[..., 1] - t.OD_peak) / t.OD_peak) ** 2
    i, j = np.unravel_index(np.argmin(err), err.shape)
    return dict(D_px=D_GRID[i], p_mm=P_GRID[j], d_mm=D_GRID[i] * P_GRID[j], M_sim=S[i, j, 0], peak_sim=S[i, j, 1], rel_err=float(np.sqrt(err[i, j])))


sim = G.Sim(os_=8, n_px=12)
rows = []
for m, t in tgt.iterrows():
    r = fit(sim, 'SUS304', t); rows.append(dict(machine=m, material='SUS304', kvp=G.KVP, sigma=t.sigma, M=t.M, OD_peak=t.OD_peak, **r))
C = pd.DataFrame(rows); C.to_csv(f'{OUT}/calibration.csv', index=False)
print(C.round(4).to_string(index=False))
# 민감도
sens = []
for kvp in (50, 90):
    sim.set_spectrum(kvp)
    for m, t in tgt.iterrows():
        sens.append(dict(machine=m, material='SUS304', kvp=kvp, **fit(sim, 'SUS304', t)))
sim.set_spectrum(G.KVP)
for mat in ('Fe', 'glass'):
    for m, t in tgt.iterrows():
        sens.append(dict(machine=m, material=mat, kvp=G.KVP, **fit(sim, mat, t)))
Sx = pd.DataFrame(sens); Sx.to_csv(f'{OUT}/sensitivity.csv', index=False)
print(Sx.round(4).to_string(index=False))
