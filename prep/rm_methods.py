"""마커 제거 방법 (g: uint8 흑백, m: uint8 마스크 1=제거할 픽셀) — build_marker_variants.py, 흔적 검사에서 공용"""
import numpy as np
import cv2
from skimage.restoration import inpaint_biharmonic

KER = np.ones((3, 3), np.uint8)


def rm_telea(g, m):  return cv2.inpaint(g, m, 3, cv2.INPAINT_TELEA)     # 기존 data/clean
def rm_mask(g, m):
    out = g.copy(); out[m > 0] = 128; return out                          # 기존 data/masked
def rm_ns(g, m):     return cv2.inpaint(g, m, 3, cv2.INPAINT_NS)
def rm_dil(g, m):    return cv2.inpaint(g, cv2.dilate(m, KER, iterations=2), 3, cv2.INPAINT_TELEA)
def rm_bgfill(g, m):
    valid = (m == 0).astype(np.float32); gf = g.astype(np.float32)
    num = cv2.GaussianBlur(gf * valid, (0, 0), 3); den = cv2.GaussianBlur(valid, (0, 0), 3)
    out = gf.copy(); out[m > 0] = (num / np.maximum(den, 1e-6))[m > 0]
    return np.clip(np.round(out), 0, 255).astype(np.uint8)
def rm_biharm(g, m):
    out = inpaint_biharmonic(g.astype(np.float64) / 255.0, m.astype(bool))
    return np.clip(np.round(out * 255), 0, 255).astype(np.uint8)


METHODS = {'rm_ns': rm_ns, 'rm_dil': rm_dil, 'rm_bgfill': rm_bgfill, 'rm_biharm': rm_biharm}
# 흔적 검사 모드 → (입력 폴더, 가짜 링을 지우는 방법)
TRACE = {'inpaint': ('clean', rm_telea), 'mask': ('masked', rm_mask), **{k: (k, f) for k, f in METHODS.items()}}
