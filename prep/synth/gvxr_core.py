"""gVXR(gVirtualXRay 2.1) 기반 이물 합성기 핵심 — 실제 KAMP 배경은 그대로 두고 이물 부분만 물리 시뮬레이션
역할 분담:
  gVXR  : 3D 형상(구·정육면체·판·선·불규칙 조각)의 투영 두께(path length) 맵, 재질별 선감쇠계수 μ(E) (원소·화합물·혼합물 데이터베이스)
  numpy : 다색 스펙트럼 투과율 T = Σ_E w(E)·exp(-μ(E)·L) / Σ_E w(E), 검출기 흐림(투과율에 가우시안), 픽셀 적분, 광학 밀도 OD = -ln T
  삽입  : 실제 배경에 곱셈 (I = I_bg · exp(-OD), synth_core.insert 와 같음) — 제품이 이미 거른 스펙트럼 변화(빔 경화)는 무시
좌표 단위: gVXR 안에서는 '검출기 픽셀 = 1 mm' 로 놓고 형상을 픽셀 단위로 만든다 (실제 픽셀 크기를 모르므로).
          물리 두께로 바꿀 때만 p (mm/px) 를 곱한다: L_mm = L_px · p. p 는 gvxr_calibrate.py 가 실제 결함으로 정함
장비 가정 (실제 값을 모름 → 보고서에 가정으로 명시, 민감도는 gvxr_calibrate.py 에서 확인):
  관전압 70 kVp, 텅스텐 양극 Kramers 스펙트럼, 알루미늄 2 mm 여과, 에너지 적분형 검출기(신호 ∝ 광자 수 × 에너지), 평행빔
"""
import ctypes, glob, os
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import ConvexHull

K = '/data/knhyun/KAMP'
for _f in sorted(glob.glob(f'{K}/third_party/gvxr_libs/root/usr/lib/x86_64-linux-gnu/*.so.*')):   # 시스템에 없는 xcb 라이브러리 (apt 패키지에서 추출)
    if not os.path.islink(_f):
        ctypes.CDLL(_f, mode=ctypes.RTLD_GLOBAL)
from gvxrPython3 import gvxr   # noqa: E402

# 재질: (종류, 기호, 무게비, 밀도 g/cm³)
MATERIALS = {
    'SUS304': ('mixture', ['Fe', 'Cr', 'Ni', 'Mn'], [0.70, 0.19, 0.09, 0.02], 8.00),    # 스테인리스 (검사기 표준 시편에 흔한 재질)
    'Fe': ('element', 'Fe', None, 7.874),
    'Al': ('element', 'Al', None, 2.699),
    'glass': ('compound', 'SiO2', None, 2.50),                                          # 유리 (소다석회 유리를 SiO2 로 근사)
    'stone': ('compound', 'CaCO3', None, 2.71),                                         # 돌 (석회석)
    'bone': ('mixture', ['H', 'C', 'N', 'O', 'Na', 'Mg', 'P', 'S', 'Ca'],
             [.034, .155, .042, .435, .001, .002, .103, .003, .225], 1.92),            # 뼈 (ICRU 피질골)
    'plastic': ('compound', 'C3H6', None, 0.90),                                        # 플라스틱 (폴리프로필렌)
}
KVP, AL_MM, E_MIN = 70.0, 2.0, 10


class Sim:
    def __init__(self, os_=8, n_px=24, kvp=KVP, al_mm=AL_MM):
        self.os, self.n = os_, n_px
        gvxr.createOpenGLContext()
        gvxr.setSourcePosition(0, 0, -1000, 'mm'); gvxr.useParallelBeam()
        gvxr.setMonoChromatic(60, 'keV', 1)
        gvxr.setDetectorPosition(0, 0, 100, 'mm'); gvxr.setDetectorUpVector(0, 1, 0)
        self.resize(n_px)
        self.E = np.arange(E_MIN, 120) + 0.5                              # μ 표는 넉넉히 (120 keV 까지), 스펙트럼 밖은 가중치 0
        self.mu = {m: self._mu(m) for m in MATERIALS}                     # 1/mm
        self.set_spectrum(kvp, al_mm)

    def set_spectrum(self, kvp, al_mm=AL_MM):
        n_ph = np.clip(kvp - self.E, 0, None) / self.E                    # Kramers 광자 수
        self.w = n_ph * np.exp(-self.mu['Al'] * al_mm) * self.E           # 여과 후, 에너지 적분 검출기
        self.w /= self.w.sum(); self.kvp = kvp
        self.ke = self.w > 0                                               # 계산에 쓰는 에너지 구간 (관전압 미만)

    def resize(self, n_px):
        self.n = n_px
        gvxr.setDetectorNumberOfPixels(n_px * self.os, n_px * self.os)
        gvxr.setDetectorPixelSize(1.0 / self.os, 1.0 / self.os, 'mm')     # '1 mm' = 검출기 1 픽셀

    def _mu(self, m):
        kind, sym, wt, rho = MATERIALS[m]
        lab = f'mu_{m}'
        gvxr.makeCube(lab, 1, 'mm')
        if kind == 'element':
            gvxr.setElement(lab, sym)
        elif kind == 'compound':
            gvxr.setCompound(lab, sym); gvxr.setDensity(lab, rho, 'g/cm3')
        else:
            gvxr.setMixture(lab, sym, wt); gvxr.setDensity(lab, rho, 'g/cm3')
        mu = np.array([gvxr.getLinearAttenuationCoefficient(lab, float(e), 'keV') for e in self.E]) / 10.0   # 1/cm → 1/mm
        gvxr.removePolygonMeshesFromSceneGraph()
        return mu

    def mu_eff(self, m, L_mm=0.0):
        """두께 L_mm 를 지난 뒤의 실효 감쇠계수 (빔 경화 반영): -d ln T / dL"""
        a = self.w * np.exp(-self.mu[m] * L_mm)
        return float((a * self.mu[m]).sum() / a.sum())

    def path_length(self, shape, size, rot=(0, 0, 0), off=(0.0, 0.0), seed=0):
        """형상의 투영 두께 맵 (부분 픽셀 해상도, 픽셀 단위). size = 형상별 크기(px), off = 중심의 부분 픽셀 이동(px)
        sphere: size = 지름 / cube: 한 변 / plate: (한 변, 두께) / wire: (지름, 길이) / irregular: 등가 지름(같은 부피의 구)"""
        gvxr.removePolygonMeshesFromXRayRenderer(); gvxr.removePolygonMeshesFromSceneGraph()
        lab = 'obj'
        if shape == 'sphere':
            gvxr.makeSphere(lab, 48, 48, size / 2, 'mm')
        elif shape == 'cube':
            gvxr.makeCube(lab, size, 'mm')
        elif shape == 'plate':
            gvxr.makeCuboid(lab, size[0], size[0], size[1], 'mm')
        elif shape == 'wire':
            gvxr.makeCylinder(lab, 32, size[1], size[0] / 2, 'mm')
        elif shape == 'irregular':   # 무작위 볼록 다면체 (돌·유리 조각), 부피를 등가 지름의 구에 맞춤
            rng = np.random.default_rng(seed)
            pts = rng.normal(size=(14, 3)) * rng.uniform(0.6, 1.4, 3)
            hull = ConvexHull(pts); pts = pts * ((np.pi / 6 * size ** 3) / hull.volume) ** (1 / 3)
            tri = hull.simplices.copy()
            c = pts.mean(0)
            for t in tri:   # 바깥쪽을 보도록 감는 방향 정렬
                n_ = np.cross(pts[t[1]] - pts[t[0]], pts[t[2]] - pts[t[0]])
                if np.dot(n_, pts[t[0]] - c) < 0:
                    t[1], t[2] = t[2], t[1]
            gvxr.makeTriangularMesh(lab, (pts - c).ravel().tolist(), tri.ravel().tolist(), 'mm')
        gvxr.addPolygonMeshAsInnerSurface(lab); gvxr.setElement(lab, 'Fe')
        if any(rot):
            gvxr.rotateNode(lab, float(rot[0]), 0, 0, 1); gvxr.rotateNode(lab, float(rot[1]), 0, 1, 0); gvxr.rotateNode(lab, float(rot[2]), 1, 0, 0)
        gvxr.translateNode(lab, float(-off[0]), float(-off[1]), 0, 'mm')   # 검출기 가로축·세로축이 영상 x·y 와 반대 (확인: 무게중심 = 5.5 + off)
        L = np.array(gvxr.computePathLength(lab), dtype=np.float64)      # gVXR 단위: cm → mm(= px) 로 환산은 아래에서 확인
        return np.flipud(L) * self.L_SCALE

    L_SCALE = 10.0   # computePathLength 출력(cm) → mm(= px). gvxr_calibrate.py 의 지름 검사로 확인

    def od_maps_p(self, L_sub, material, p_grid, sigma_px):
        """od_map 을 픽셀 크기 여러 개에 대해 한 번에 (보정용). 반환 (len(p_grid), n, n)"""
        L = np.multiply.outer(np.asarray(p_grid), L_sub)                       # (P, H, W) mm
        T = np.tensordot(self.w[self.ke], np.exp(-np.multiply.outer(self.mu[material][self.ke], L)), axes=1)   # (P, H, W)
        if sigma_px > 0:
            T = 1.0 - gaussian_filter(1.0 - T, (0, sigma_px * self.os, sigma_px * self.os), mode='constant')
        n, o = self.n, self.os
        Tp = T.reshape(len(p_grid), n, o, n, o).mean((2, 4))
        return -np.log(np.clip(Tp, 1e-6, 1.0))

    def od_map(self, L_sub, material, p_mm, sigma_px):
        """투영 두께(부분 픽셀, px) → 다색 투과율 → 검출기 흐림(투과율에 σ px 가우시안) → 픽셀 적분 → OD 맵 (n×n px)"""
        L = L_sub * p_mm
        T = np.tensordot(self.w[self.ke], np.exp(-np.multiply.outer(self.mu[material][self.ke], L)), axes=1)
        if sigma_px > 0:
            T = gaussian_filter(1.0 - T, sigma_px * self.os, mode='constant')   # 1 - T 를 흐림 (바깥은 0) → 경계 처리 단순
            T = 1.0 - T
        n, o = self.n, self.os
        Tp = T.reshape(n, o, n, o).mean((1, 3))
        return -np.log(np.clip(Tp, 1e-6, 1.0)).astype(np.float32)


def make_object(sim, shape, d_eq, material, p_mm, sigma, rng):
    """이물 하나의 광학 밀도 패치. d_eq = 같은 부피 구의 지름(px). shape: sphere / cube / irregular / plate / wire<r>(길이 = 지름 × r)
    반환 (od 패치 n×n, 부분 픽셀 이동 off). 물체 중심 = 패치 좌표 n/2 - 0.5 + off"""
    V = np.pi / 6 * d_eq ** 3
    rot = tuple(rng.uniform(0, 360, 3)); off = tuple(rng.uniform(0, 1, 2))
    if shape == 'sphere':
        args, ext = ('sphere', d_eq), d_eq
    elif shape == 'cube':
        a = V ** (1 / 3); args, ext = ('cube', a), a * 1.8
    elif shape == 'irregular':
        args, ext = ('irregular', d_eq), d_eq * 2.2
    elif shape == 'plate':
        a = (5 * V) ** (1 / 3); args, ext = ('plate', (a, a / 5)), a * 1.5
    else:
        r_ = float(shape[4:]); d = (4 * V / (np.pi * r_)) ** (1 / 3); args, ext = ('wire', (d, d * r_)), d * r_ + 2
    n = int(np.ceil(ext + 6 * sigma + 6)); n += n % 2
    sim.resize(max(n, 12))
    L = sim.path_length(args[0], args[1], rot=rot, off=off, seed=int(rng.integers(1 << 30)))
    return sim.od_map(L, material, p_mm, sigma), off
