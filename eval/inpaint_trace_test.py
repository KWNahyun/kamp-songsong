"""인페인팅 흔적 shortcut 검사
결함이 없는 제품 영역에 실제 마커와 같은 형태(18×18, 두께 2px)의 링을 인페인팅해 넣고,
그 위치에서 모델 검출이 생기는지를 원본(대조군)과 비교한다.
출력: eval/out/inpaint_trace_sites.csv, 콘솔 요약
"""
import os, sys
import numpy as np
import pandas as pd
import cv2
import torch

K = '/data/knhyun/KAMP'
sys.path.insert(0, f'{K}/eval')
# 사용: inpaint_trace_test.py [--mode inpaint|mask] [run ...]
#   inpaint: 인페인팅 입력(data/clean)에 가짜 링을 Telea 로 지움 / mask: 마스킹 입력(data/masked)에 가짜 링을 회색 128 로 칠함
#   rm_ns|rm_dil|rm_bgfill|rm_biharm: 해당 제거본 입력에 가짜 링을 같은 방법으로 지움
#   환경변수 NB2_MODEL=yolov8s|dfine_s: 새 베이스라인 seed 3개 (640)로 검사 (run 인자 무시)
args = sys.argv[1:]
MODE = 'inpaint'
if args[:1] == ['--mode']:
    MODE, args = args[1], args[2:]
RUNS = args or ['B_clean416', 'C_clean640']
sys.path.insert(0, f'{K}/prep')
from rm_methods import TRACE
BASE, FILL = TRACE[MODE]
sys.argv = ['x']
exec(open(f'{K}/eda/extract.py').read().split('# ─────────────────────────── 1.')[0])   # product_mask 재사용
import evaluate as E

RNG = np.random.default_rng(0)
N_SITES, MIN_GAP, RING, THICK, MATCH = 3, 30, 18, 2, 8.0
CONFS = [0.01, 0.05, 0.1, 0.25]
split = pd.read_csv(f'{K}/data/splits/split.csv')
stems = split[split.split.isin(['val', 'test'])].stem.tolist()


def ring_mask(shape, cx, cy):
    m = np.zeros(shape, np.uint8)
    x0, y0 = cx - RING // 2, cy - RING // 2
    cv2.rectangle(m, (x0, y0), (x0 + RING - 1, y0 + RING - 1), 1, THICK)
    return m


def gt_centers(stem, W, H):
    g = E.load_gt('clean', stem, W, H)
    return ((g[:, :2] + g[:, 2:]) / 2).numpy()


def pick_sites(clean, gts):
    pm = product_mask(clean)
    ys, xs = np.where(cv2.erode(pm.astype(np.uint8), np.ones((RING + 4, RING + 4), np.uint8)) > 0)
    sites = []
    for _ in range(2000):
        if len(sites) == N_SITES or not len(xs):
            break
        i = RNG.integers(len(xs)); x, y = xs[i], ys[i]
        if len(gts) and np.hypot(gts[:, 0] - x, gts[:, 1] - y).min() < MIN_GAP:
            continue
        if any(np.hypot(sx - x, sy - y) < MIN_GAP for sx, sy in sites):
            continue
        sites.append((int(x), int(y)))
    return sites


def dets_near(det, x, y):
    if not len(det):
        return 0.0
    c = (det[:, :2] + det[:, 2:4]) / 2
    d = torch.hypot(c[:, 0] - x, c[:, 1] - y)
    near = det[d <= MATCH]
    return float(near[:, 4].max()) if len(near) else 0.0


rows = []
_cfg = {r: (w, s) for r, w, s in E.MODELS}
_cfg_of = lambda r: _cfg.get(r, (f'{K}/runs/{r}/weights/best.pt', 416))   # 목록에 없으면 runs/<이름>
if os.environ.get('NB2_MODEL'):
    import nb2_models as NB
    NB.patch(E); _sp = NB.specs(os.environ['NB2_MODEL']); RUNS = list(_sp)
    _cfg_of = lambda r: (_sp[r], 640)
models = {r: E.load_model(_cfg_of(r)[0]) for r in RUNS}
sizes = {r: _cfg_of(r)[1] for r in RUNS}
for stem in stems:
    path = f'{K}/data/{BASE}/images/{stem}.png'
    clean = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    site_img = cv2.imread(f'{K}/data/clean/images/{stem}.png', cv2.IMREAD_GRAYSCALE)   # 위치 선정은 두 모드 공통
    H, W = clean.shape
    gts = gt_centers(stem, W, H)
    sites = pick_sites(site_img, gts)
    mask = np.zeros_like(clean)
    for x, y in sites:
        mask |= ring_mask(clean.shape, x, y)
    traced = FILL(clean, mask)
    for r, m in models.items():
        _, det0 = E.predict(m, path, sizes[r])      # 대조군: 원본(마커 제거본)
        _, det1 = E.predict_img(m, traced, sizes[r])   # 가짜 링 삽입본 (메모리에서 바로 추론: 임시 파일 공유로 인한 병렬 실행 충돌 방지)
        # 실제 결함 검출이 흔적 삽입으로 달라졌는지도 확인
        for x, y in sites:
            rows.append(dict(run=r, stem=stem, machine=split.set_index('stem').machine[stem], x=x, y=y,
                             conf_control=dets_near(det0, x, y), conf_traced=dets_near(det1, x, y),
                             changed_px=int((traced != clean).sum())))
df = pd.DataFrame(rows)
TRACE_OUT = os.environ.get('TRACE_OUT', f'{K}/eval/out'); os.makedirs(TRACE_OUT, exist_ok=True)
df.to_csv(f'{TRACE_OUT}/trace_sites_{MODE}.csv', index=False)

print(f'[{MODE}] 이미지 {len(stems)}장, 모델당 가짜 링 {len(df) // len(models)}개')
out = []
for r, g in df.groupby('run'):
    for c in CONFS:
        out.append(dict(run=r, conf_thr=c, control_hit=(g.conf_control >= c).mean(), traced_hit=(g.conf_traced >= c).mean(),
                        n_ctrl=(g.conf_control >= c).sum(), n_trace=(g.conf_traced >= c).sum()))
res = pd.DataFrame(out)
print(res.round(4).to_string(index=False))
print('\n링 위치 최대 conf (대조군 / 삽입본):')
print(df.groupby('run')[['conf_control', 'conf_traced']].max().round(4))
res.to_csv(f'{TRACE_OUT}/trace_summary_{MODE}.csv', index=False)
