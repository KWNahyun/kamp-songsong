"""합성 결함 핵심 함수 (가설: Beer-Lambert 곱셈 감쇠) — 검증(eval/synth/validate_replica.py)을 통과하기 전에는 학습에 쓰지 않는다

모델 (마커 제거 영상 I, 결함 없는 배경 B):
  I_syn(x) = ( B(x) + n(x) ) · T(x),   T(x) = exp( -OD · g(x) )
    OD(x) = OD0 · box(x),  box = 사각 물체(한 변 w) ⊗ 장비 흐림(가우시안 σ), 픽셀 면적으로 적분 (결함이 픽셀 크기라 점 샘플링 대신 적분)
      OD0 = 물체 내부의 광학 밀도 (재질 밀도 × 두께). 크기(w)·흐림(σ)·밀도(OD0)를 따로 조절할 수 있음
    (시도 1: 가우시안 점 OD = M·gauss(σ) → 검증 실패. 실제 결함보다 넓게 번지고 중심이 더 어두움. eval/synth/attempt1_gauss/)
    n(x) = 지운 영역에만 채우는 잡음 (지운 영역은 인페인팅으로 매끈해져 원래 잡음이 사라지므로 복원)
비교용 변형 (검증에서 '틀린 합성'을 검출할 수 있는지 확인하는 양성 대조):
  add   : I_syn = B + n - C·g        (덧셈형: 배경과 무관하게 같은 밝기만큼 어두워짐)
  sharp : g = 2×2 사각 점 (흐림 없음)
  nonoise: n = 0 (지운 영역 잡음 복원 안 함)
"""
import numpy as np
import cv2
from scipy import ndimage as ndi
from scipy.optimize import least_squares
from scipy.special import erf

KER3 = np.ones((3, 3), np.uint8)


def core_mask(img, px, py, bg, contrast, grow=2, win=4):
    """실제 결함의 지울 영역: 결함 중심과 연결된 '배경 - 대비/2 보다 어두운' 픽셀 + grow px 확장"""
    H, W = img.shape
    y0, y1, x0, x1 = max(py - win, 0), min(py + win + 2, H), max(px - win, 0), min(px + win + 2, W)
    w = img[y0:y1, x0:x1].astype(np.float32)
    dark = w < (bg - contrast / 2)
    lab, _ = ndi.label(dark)
    cy, cx = py - y0, px - x0
    ids = [lab[yy, xx] for yy in (cy, cy + 1) for xx in (cx, cx + 1) if 0 <= yy < lab.shape[0] and 0 <= xx < lab.shape[1] and lab[yy, xx]]
    m = np.zeros((H, W), np.uint8)
    if ids:
        m[y0:y1, x0:x1] = np.isin(lab, ids)
    else:   # 반치 영역을 못 찾으면 2×2 코어
        m[py:py + 2, px:px + 2] = 1
    return cv2.dilate(m, KER3, iterations=grow)


def erase(img, mask):
    """Telea 인페인팅 (마커 제거와 같은 방법)"""
    return cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)


def refill_noise(bg_img, mask, sigma, rng):
    out = bg_img.astype(np.float32)
    out[mask > 0] += rng.normal(0, sigma, int(mask.sum()))
    return out


def _pix1d(n, c, s):
    e = np.arange(n + 1) - 0.5                       # 픽셀 경계
    cdf = 0.5 * (1 + erf((e - c) / (np.sqrt(2) * s)))
    return np.diff(cdf)


def _ipsi(t):
    """∫Φ(t)dt = tΦ(t) + φ(t)"""
    return t * 0.5 * (1 + erf(t / np.sqrt(2))) + np.exp(-t ** 2 / 2) / np.sqrt(2 * np.pi)


def _box1d(n, c, w, s):
    """폭 w 의 상자(중심 c)를 가우시안 σ 로 흐린 뒤 픽셀 [i-.5, i+.5] 로 적분한 값 (상자 내부 = 1 기준, 픽셀 면적 1)"""
    e = np.arange(n + 1) - 0.5
    a, b = c - w / 2, c + w / 2
    F = s * (_ipsi((e - a) / s) - _ipsi((e - b) / s))     # ∫[Φ((u-a)/σ) - Φ((u-b)/σ)] du
    return np.diff(F)


def box(shape, x0, y0, w, s, h=None):
    """사각 물체(가로 w, 세로 h) ⊗ 가우시안 흐림 σ, 픽셀 적분. 물체 내부 = 1"""
    return np.outer(_box1d(shape[0], y0, w if h is None else h, s), _box1d(shape[1], x0, w, s)).astype(np.float32)


def gauss(shape, x0, y0, s):
    """픽셀 적분 가우시안 (합 = 1)"""
    return np.outer(_pix1d(shape[0], y0, s), _pix1d(shape[1], x0, s)).astype(np.float32)


def square2(shape, x0, y0):
    g = np.zeros(shape, np.float32)
    xi, yi = int(np.floor(x0)), int(np.floor(y0))
    g[yi:yi + 2, xi:xi + 2] = 1
    return g


def fit_box(real, erased, px, py, win=4):
    """관측 광학 밀도 -ln(real/erased) 에 OD0·box(w, σ) 를 맞춤 → OD0(물체 내부 광학 밀도), w(물체 한 변), σ(장비 흐림), x0, y0"""
    H, W = real.shape
    y0, y1, x0, x1 = max(py - win, 0), min(py + win + 2, H), max(px - win, 0), min(px + win + 2, W)
    r = real[y0:y1, x0:x1].astype(np.float64); e = np.maximum(erased[y0:y1, x0:x1].astype(np.float64), 1)
    od_obs = -np.log(np.clip(r / e, 1e-3, 1.5)); sh = od_obs.shape

    def model(p):
        od0, cx, cy, w, s = p
        return od0 * box(sh, cx - x0, cy - y0, w, s)

    def res(p):
        return (model(p) - od_obs).ravel()
    best = None
    for w0 in (1.5, 2.0, 2.5):      # 초기값 여러 개 중 가장 잘 맞는 것
        p0 = [float(np.clip(od_obs.max(), .1, 4.9)), px + 1, py + 1, w0, .4]
        f = least_squares(res, p0, bounds=([0, px - 3, py - 3, .5, .05], [5, px + 4, py + 4, 8, 2]))
        if best is None or f.cost < best.cost:
            best = f
    od0, cx, cy, w, s = best.x
    out = dict(OD0=od0, x0=cx, y0=cy, w=w, sigma=s, M=od0 * w * w, OD_peak=float(model(best.x).max()),
               OD_obs_peak=float(od_obs.max()), fit_rmse=float(np.sqrt(np.mean(best.fun ** 2))))
    # 격자 정렬: 실제 결함은 픽셀 격자에 맞춘 2×2 블록이 많음 → 중심을 가장 가까운 픽셀 경계(정수 + 0.5)로 고정하고 OD0·w·σ 재적합
    sx, sy = np.floor(cx) + 0.5, np.floor(cy) + 0.5
    f2 = least_squares(lambda q: res([q[0], sx, sy, q[1], q[2]]), [od0, w, s], bounds=([0, .5, .05], [5, 8, 2]))
    out.update(sn_OD0=f2.x[0], sn_x0=sx, sn_y0=sy, sn_w=f2.x[1], sn_sigma=f2.x[2], sn_fit_rmse=float(np.sqrt(np.mean(f2.fun ** 2))))
    return out


def fit_spot(real, erased, px, py, win=4):
    """실제 결함의 관측 광학 밀도 -ln(real/erased) 에 M·gauss 를 맞춤 → M, x0, y0, σ (전체 영상 좌표), OD_peak"""
    H, W = real.shape
    y0, y1, x0, x1 = max(py - win, 0), min(py + win + 2, H), max(px - win, 0), min(px + win + 2, W)
    r = real[y0:y1, x0:x1].astype(np.float64); e = np.maximum(erased[y0:y1, x0:x1].astype(np.float64), 1)
    od_obs = -np.log(np.clip(r / e, 1e-3, 1.5))          # 관측 광학 밀도 맵
    sh = od_obs.shape

    def model(p):
        m, cx, cy, s = p
        return m * gauss(sh, cx - x0, cy - y0, s)

    def res(p):
        return (model(p) - od_obs).ravel()
    p0 = [float(np.clip(od_obs[od_obs > 0].sum(), .1, 19)), px + .5, py + .5, .7]
    f = least_squares(res, p0, bounds=([0, px - 3, py - 3, .2], [20, px + 4, py + 4, 4]))
    m, cx, cy, s = f.x
    return dict(M=m, x0=cx, y0=cy, sigma=s, OD_peak=float(model(f.x).max()), OD_obs_peak=float(od_obs.max()),
                fit_rmse=float(np.sqrt(np.mean(f.fun ** 2))))


def insert(bg_f, g, OD=None, C=None, mode='mult'):
    """bg_f: float 배경(잡음 복원 포함). mult: bg·exp(-OD g) / add: bg - C g"""
    out = bg_f * np.exp(-OD * g) if mode == 'mult' else bg_f - C * g
    return np.clip(np.round(out), 0, 255).astype(np.uint8)
