"""X-ray 이물질 데이터셋 심층 분석 — 통계 추출
출력: eda/out/{raw_inventory,images,objects,markers}.csv
데이터 출처: 장비 원본 BMP 폴더 test1/yolov3/X선이물검출기(06.23_09.22)/ (고유 2,532장)
  - GT 데이터 500장: 공식 TXT 라벨(라벨링 6종 세트/labels)이 있는 이미지. 그중 400장은 images 400 과 바이트 동일
  - 미라벨 2,020장: 나머지에서 튜토리얼용 12장(test1/yolov3/images, OpenLabeling input)을 제외
"""
import os, re, glob, hashlib
from collections import defaultdict
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from scipy import ndimage as ndi

ROOT = '/data/knhyun/KAMP/xray_dataset/4. X-ray 검사장비 AI 데이터셋/dataset'
SET_DIR = f'{ROOT}/라벨링 6종 세트'
RAW_DIR = f'{ROOT}/test1/yolov3/X선이물검출기(06.23_09.22)'
BASELINE_DIR = '/data/knhyun/KAMP/baseline'
OUT = '/data/knhyun/KAMP/eda/out'
os.makedirs(OUT, exist_ok=True)

NAME_RE = re.compile(r'(\d{3})_(\d{8})_(\d{6})\((\d)\)')


def parse_name(stem):
    prod, date, time, seq = NAME_RE.match(stem).groups()
    ts = pd.Timestamp(f'{date} {time[:2]}:{time[2:4]}:{time[4:]}')
    return prod, date, ts, int(seq)


def load_rgb(path):
    """팔레트(P) 이미지 → (gray, colored_mask). 컬러 픽셀 = 장비가 새긴 NG 마커."""
    im = Image.open(path)
    idx = np.array(im)
    pal = np.array(im.getpalette()).reshape(-1, 3)
    rgb = pal[idx]
    colored = ~((rgb[..., 0] == rgb[..., 1]) & (rgb[..., 1] == rgb[..., 2]))
    gray = rgb[..., 0].astype(np.float32)  # 회색 픽셀은 R=G=B
    return gray, colored, rgb


def remove_markers(gray, colored):
    """마커 픽셀을 주변 회색값으로 inpaint"""
    g8 = gray.clip(0, 255).astype(np.uint8)
    return cv2.inpaint(g8, colored.astype(np.uint8), 3, cv2.INPAINT_TELEA)


def load_labels(stem):
    p = f'{SET_DIR}/labels/{stem}.txt'
    if not os.path.exists(p):
        return np.zeros((0, 5))
    rows = [list(map(float, l.split())) for l in open(p) if l.strip()]
    return np.array(rows) if rows else np.zeros((0, 5))


def product_mask(clean):
    """제품 영역: 밝은 배경(컨베이어) 대비 어두운 최대 연결영역"""
    blur = cv2.GaussianBlur(clean, (9, 9), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(th)
    if n <= 1:
        return np.zeros_like(th, bool)
    k = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return ndi.binary_fill_holes(lab == k)


def dhash(img, size=16):
    r = cv2.resize(img, (size + 1, size), interpolation=cv2.INTER_AREA)
    return (r[:, 1:] > r[:, :-1]).flatten()


# ─────────────────────────── 1. 원본(raw) 인벤토리 ───────────────────────────
official = set(os.path.splitext(f)[0] for f in os.listdir(f'{SET_DIR}/labels'))
practice = ({os.path.splitext(os.path.basename(f))[0] for f in glob.glob(f'{ROOT}/test1/yolov3/images/*')} |
            {os.path.splitext(os.path.basename(f))[0] for f in glob.glob(f'{ROOT}/OpenLabeling-master/main/input/*')}) - official
raw_rows = []
for p in sorted(glob.glob(f'{RAW_DIR}/*/*/*.bmp')):
    stem = os.path.splitext(os.path.basename(p))[0]
    prod, date, ts, seq = parse_name(stem)
    im = Image.open(p)
    raw_rows.append(dict(stem=stem, machine=p.split('/')[-3][:3], folder_date=p.split('/')[-2].split('_')[1],
                         product=prod, date=date, ts=ts, w=im.size[0], h=im.size[1],
                         md5=hashlib.md5(open(p, 'rb').read()).hexdigest(), path=p,
                         is_gt=stem in official, practice=stem in practice))
raw = pd.DataFrame(raw_rows)
raw['use'] = np.where(raw.is_gt, 'GT', np.where(raw.practice, '제외(튜토리얼)', '미라벨'))
raw.to_csv(f'{OUT}/raw_inventory.csv', index=False)
stem2machine = raw.groupby('stem').machine.first().to_dict()
stem2rawpath = raw.groupby('stem').path.first().to_dict()
u = raw.drop_duplicates('md5')
print(f'원본 파일 {len(raw)}개 → 고유 이미지 {len(u)}장 | GT {u.is_gt.sum()} / 미라벨 {(u.use == "미라벨").sum()} / 제외(튜토리얼) {u.practice.sum()}')
assert official <= set(raw.stem), '원본 폴더에 없는 공식 라벨 존재'

# ─────────────────────────── 2. GT 이미지 (공식 라벨 500장) ───────────────────────────
subsets = {s: set(os.path.splitext(f)[0] for f in os.listdir(f'{SET_DIR}/images {s}'))
           for s in [15, 50, 100, 200, 300, 400]}
split_of = {}
for sp in ['train', 'val', 'test']:
    for l in open(f'{BASELINE_DIR}/{sp}.txt'):
        split_of[os.path.splitext(os.path.basename(l.strip()))[0]] = sp

label_stems = sorted(official)
# images 400 은 원본 BMP 의 바이트 동일 복사본임을 확인 (출처를 원본 폴더로 통일)
md5_of = raw.groupby('stem').md5.apply(set).to_dict()
assert all(hashlib.md5(open(f'{SET_DIR}/images 400/{s}.jpg', 'rb').read()).hexdigest() in md5_of[s] for s in subsets[400])
img_rows, obj_rows, mk_rows = [], [], []

for stem in label_stems:
    path = stem2rawpath[stem]
    in400 = stem in subsets[400]
    if path is None:
        continue
    gray, colored, rgb = load_rgb(path)
    H, W = gray.shape
    clean = remove_markers(gray, colored)
    pmask = product_mask(clean)
    dist_in = cv2.distanceTransform(pmask.astype(np.uint8), cv2.DIST_L2, 5)
    prod, date, ts, seq = parse_name(stem)
    labels = load_labels(stem)

    # ── 마커(컬러 사각형) 연결요소
    lab_mk, n_mk = ndi.label(colored, structure=np.ones((3, 3)))
    mk_boxes = ndi.find_objects(lab_mk)
    colors = defaultdict(int)
    for sl_i, sl in enumerate(mk_boxes, 1):
        comp = lab_mk[sl] == sl_i
        cols = rgb[sl][comp]
        uniq, cnt = np.unique(cols, axis=0, return_counts=True)
        cname = '+'.join(sorted({ {(255, 0, 0): 'red', (0, 0, 255): 'blue', (255, 0, 255): 'magenta',
                                   (255, 255, 0): 'yellow', (0, 255, 0): 'green'}.get(tuple(u), str(tuple(u))) for u in uniq}))
        colors[cname] += 1
        y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        n_gt_inside = sum(1 for b in labels if x0 <= b[1] * W <= x1 and y0 <= b[2] * H <= y1)
        mk_rows.append(dict(stem=stem, x0=x0, y0=y0, w=x1 - x0, h=y1 - y0, n_px=int(comp.sum()),
                            color=cname, n_gt_inside=n_gt_inside))

    # dHash (마커 제거본) — 근접중복 탐지용
    hsh = dhash(clean)

    img_rows.append(dict(
        stem=stem, labeled_in_400=in400, machine=stem2machine[stem], product=prod,
        date=date, ts=ts, seq=seq, W=W, H=H, n_obj=len(labels),
        split=split_of.get(stem, 'extra100' if not in400 else 'NA'),   # 기존 베이스라인 무작위 분할 (누수 분석용)
        min_subset=min([s for s in subsets if stem in subsets[s]], default=None),
        n_colored_px=int(colored.sum()), n_marker_cc=n_mk,
        bg_level=float(np.median(clean[~pmask])) if (~pmask).any() else np.nan,
        prod_level=float(np.median(clean[pmask])) if pmask.any() else np.nan,
        prod_area_frac=float(pmask.mean()),
        img_noise=float(np.median(np.abs(clean[~pmask] - np.median(clean[~pmask]))) * 1.4826) if (~pmask).any() else np.nan,
        dhash=''.join('1' if b else '0' for b in hsh),
    ))

    # ── 객체 단위 통계
    for oi, (c, cx, cy, bw, bh) in enumerate(labels):
        x0 = int(round((cx - bw / 2) * W)); x1 = int(round((cx + bw / 2) * W))
        y0 = int(round((cy - bh / 2) * H)); y1 = int(round((cy + bh / 2) * H))
        x0c, y0c, x1c, y1c = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
        # 결함 코어: GT 박스 내부에서 2x2 평균이 가장 어두운 지점 (결함 본체가 2~3px 점)
        cf = clean.astype(np.float32)
        box2 = cv2.blur(cf, (2, 2), anchor=(0, 0), borderType=cv2.BORDER_REPLICATE)
        inner2 = box2[y0c:y1c, x0c:x1c]
        my, mx = np.unravel_index(np.argmin(inner2), inner2.shape)
        py, px = y0c + my, x0c + mx          # 2x2 블록 좌상단
        core_val = float(inner2[my, mx])
        dark_dx = (px + 1.0) - cx * W
        dark_dy = (py + 1.0) - cy * H
        # 로컬 배경: 결함 중심 9x9 창에서 중앙 5x5 제외 영역의 median
        def win(a, r):
            return a[max(py - r, 0):py + r + 2, max(px - r, 0):px + r + 2]
        w9 = win(cf, 4).copy()
        cmask = np.zeros_like(w9, bool)
        cy0, cx0 = min(py, 4) , min(px, 4)
        cmask[max(cy0 - 2, 0):cy0 + 4, max(cx0 - 2, 0):cx0 + 4] = True
        bg = float(np.median(w9[~cmask]))
        # 노이즈: 고주파 잔차(원본 - median5) 의 MAD, 15x15 창 (구조물 경계 영향 제거)
        resid = cf - cv2.medianBlur(clean, 5).astype(np.float32)
        sigma = float(np.median(np.abs(win(resid, 7))) * 1.4826)
        contrast = bg - core_val
        # 반치폭(FWHM) 면적: 결함 중심과 연결된, bg - contrast/2 보다 어두운 픽셀 수
        w7 = win(cf, 3)
        dark = w7 < (bg - contrast / 2)
        lab_d, _ = ndi.label(dark)
        c_id = lab_d[min(py, 3), min(px, 3)] or lab_d[min(py, 3) + 1, min(px, 3) + 1]
        core_px = int((lab_d == c_id).sum()) if c_id else 0
        # 배경(제품 두께/밀도 proxy): 결함 주변 21x21 median
        bg_wide = float(np.median(win(cf, 10)))
        pcx, pcy = int(np.clip(cx * W, 0, W - 1)), int(np.clip(cy * H, 0, H - 1))
        # 마커와의 관계: GT 중심을 포함하는 마커 연결요소
        mk_id = lab_mk[max(y0c - 20, 0):min(y1c + 20, H), max(x0c - 20, 0):min(x1c + 20, W)]
        obj_rows.append(dict(
            stem=stem, obj=oi, cx=cx, cy=cy, bw=bw, bh=bh,
            w_px=bw * W, h_px=bh * H, area_px=bw * W * bh * H, W=W, H=H,
            edge_dist_img=min(cx, 1 - cx, cy, 1 - cy),
            in_product=bool(pmask[pcy, pcx]),
            dist_prod_edge_px=float(dist_in[pcy, pcx]),
            bg_local=bg, bg_wide=bg_wide, noise_local=sigma, core_val=core_val,
            contrast=contrast, cnr=contrast / max(sigma, 0.5),
            core_px=core_px, dark_dx=dark_dx, dark_dy=dark_dy,
            marker_near=bool((mk_id > 0).any()),
            out_of_range=bool(x0 < 0 or y0 < 0 or x1 > W or y1 > H),
        ))

img = pd.DataFrame(img_rows)
obj = pd.DataFrame(obj_rows)
mk = pd.DataFrame(mk_rows)
obj = obj.merge(img[['stem', 'machine', 'product', 'split', 'labeled_in_400', 'n_obj', 'min_subset']], on='stem')
img.to_csv(f'{OUT}/images.csv', index=False)
obj.to_csv(f'{OUT}/objects.csv', index=False)
mk.to_csv(f'{OUT}/markers.csv', index=False)
print(f'images: {len(img)}  objects: {len(obj)}  marker comps: {len(mk)}')
