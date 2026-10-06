"""재학습 모델 평가 (세션 그룹 분할 test / val)
사용: python3 eval/evaluate.py            → 모든 (모델, 입력) 조합 평가
출력: eval/out/metrics.csv, eval/out/per_machine.csv, eval/out/dets_<run>_<variant>_<split>.csv, gts_...csv
"""
import os, sys
import numpy as np
import pandas as pd
import torch
import cv2

K = '/data/knhyun/KAMP'
sys.path.insert(0, f'{K}/yolov3')
from models import Darknet
from utils.utils import non_max_suppression, scale_coords, compute_ap, box_iou, xywh2xyxy
from utils.datasets import letterbox

OUT = f'{K}/eval/out'
os.makedirs(OUT, exist_ok=True)
CFG = f'{K}/yolov3/yolov3-spp.cfg'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CONF_MIN, NMS_IOU, IOU_T, CENTER_T = 0.001, 0.5, 0.5, 8.0   # CENTER_T: 원본 px

split = pd.read_csv(f'{K}/data/splits/split.csv').set_index('stem')

# (run 이름, 가중치, 추론 해상도)
MODELS = [
    ('A_marked416', f'{K}/runs/A_marked416/weights/best.pt', 416),
    ('B_clean416', f'{K}/runs/B_clean416/weights/best.pt', 416),
    ('C_clean640', f'{K}/runs/C_clean640/weights/best.pt', 640),
    ('D_clean416_scratch', f'{K}/runs/D_clean416_scratch/weights/best.pt', 416),   # 외부 가중치 없이 처음부터 학습
    ('E_aug416', f'{K}/runs/E_aug416/weights/best.pt', 416),   # 합성 결함 증강 학습
    ('F_masked416', f'{K}/runs/F_masked416/weights/best.pt', 416),   # 마커 마스킹(고정 회색) 입력으로 학습
    ('G_pl25', f'{K}/runs/G_pl25/weights/best.pt', 416),   # E 설정 + 의사 라벨 (교사 E, T=0.25)
    ('G_pl37', f'{K}/runs/G_pl37/weights/best.pt', 416),   # 〃 T=0.37
    ('G_pl46', f'{K}/runs/G_pl46/weights/best.pt', 416),   # 〃 T=0.46
    # 참고용: 기존 제공 가중치 (test 이미지 일부가 학습에 포함 → 오염)
    ('orig_last400', f'{K}/xray_dataset/4. X-ray 검사장비 AI 데이터셋/dataset/실습별 가중치파일/last400.pt', 416),
]


def load_model(w):
    m = Darknet(CFG).to(DEVICE)
    ck = torch.load(w, map_location=DEVICE, weights_only=False)
    m.load_state_dict(ck['model'], strict=False)
    return m.eval()


def load_gt(variant, stem, W, H):
    rows = [list(map(float, l.split())) for l in open(f'{K}/data/{variant}/labels/{stem}.txt') if l.strip()]
    b = torch.tensor(rows)[:, 1:5] if rows else torch.zeros((0, 4))
    b = xywh2xyxy(b)
    b[:, [0, 2]] *= W; b[:, [1, 3]] *= H
    return b


def predict(model, path, imgsz):
    return predict_img(model, cv2.imread(path), imgsz)   # BGR, 3ch


@torch.no_grad()
def predict_img(model, img0, imgsz):
    """img0: BGR 3채널 또는 단채널 uint8 배열"""
    if img0.ndim == 2:
        img0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
    img = letterbox(img0, new_shape=imgsz, auto=False)[0]
    img = torch.from_numpy(img[:, :, ::-1].transpose(2, 0, 1).copy()).to(DEVICE).float()[None] / 255.0
    pred = model(img)[0]
    det = non_max_suppression(pred, CONF_MIN, NMS_IOU)[0]
    if det is None or not len(det):
        return img0.shape[:2], torch.zeros((0, 5))
    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
    return img0.shape[:2], det[:, :5].cpu()


def match(det, gt):
    """conf 내림차순 greedy 매칭. IoU 기준과 중심거리 기준을 각각 계산"""
    n, m = len(det), len(gt)
    tp_iou, tp_ctr = np.zeros(n, bool), np.zeros(n, bool)
    gt_iou = np.full(m, -1); gt_ctr = np.full(m, -1)
    if n and m:
        order = np.argsort(-det[:, 4].numpy())
        iou = box_iou(det[:, :4], gt).numpy()
        dc = ((det[:, None, :2] + det[:, None, 2:4]) / 2 - (gt[None, :, :2] + gt[None, :, 2:]) / 2).norm(dim=2).numpy()
        for i in order:
            j = np.argmax(np.where(gt_iou < 0, iou[i], -1))
            if iou[i, j] >= IOU_T and gt_iou[j] < 0:
                tp_iou[i] = True; gt_iou[j] = i
            d = np.where(gt_ctr < 0, dc[i], np.inf); j = np.argmin(d)
            if d[j] <= CENTER_T:
                tp_ctr[i] = True; gt_ctr[j] = i
    return tp_iou, tp_ctr, gt_iou, gt_ctr, (box_iou(det[:, :4], gt).numpy() if n and m else np.zeros((n, m)))


def summarize(dets, n_gt, tp_col):
    """AP@0.5 + 고정 conf(0.25)/최대 F1 지점의 P, R, F1"""
    d = dets.sort_values('conf', ascending=False)
    tp = d[tp_col].values.astype(float); fp = 1 - tp
    ctp, cfp = np.cumsum(tp), np.cumsum(fp)
    rec = ctp / max(n_gt, 1); prec = ctp / np.maximum(ctp + cfp, 1e-9)
    ap = compute_ap(rec, prec) if len(d) else 0.0
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    k = int(np.argmax(f1)) if len(d) else 0
    s = d[d.conf >= 0.25]
    p25 = s[tp_col].mean() if len(s) else 0.0
    r25 = s[tp_col].sum() / max(n_gt, 1)
    return dict(AP50=ap, P_at25=p25, R_at25=r25, F1_at25=2 * p25 * r25 / max(p25 + r25, 1e-9),
                bestF1=f1[k] if len(d) else 0, bestF1_conf=d.conf.values[k] if len(d) else np.nan,
                R_max=rec[-1] if len(d) else 0)


def evaluate(run, weights, imgsz, variant, sp, model):
    stems = split[split.split == sp].index
    det_rows, gt_rows = [], []
    for s in stems:
        path = f'{K}/data/{variant}/images/{s}.png'
        (H, W), det = predict(model, path, imgsz)
        gt = load_gt(variant, s, W, H)
        tp_iou, tp_ctr, g_iou, g_ctr, iou = match(det, gt)
        for i in range(len(det)):
            det_rows.append(dict(stem=s, x1=det[i, 0].item(), y1=det[i, 1].item(), x2=det[i, 2].item(), y2=det[i, 3].item(),
                                 conf=det[i, 4].item(), tp_iou=tp_iou[i], tp_ctr=tp_ctr[i]))
        for j in range(len(gt)):
            best = iou[:, j].max() if len(det) else 0.0
            gt_rows.append(dict(stem=s, gt=j, cx=((gt[j, 0] + gt[j, 2]) / 2).item(), cy=((gt[j, 1] + gt[j, 3]) / 2).item(),
                                w=(gt[j, 2] - gt[j, 0]).item(), h=(gt[j, 3] - gt[j, 1]).item(),
                                conf_iou=det[g_iou[j], 4].item() if g_iou[j] >= 0 else 0.0,
                                conf_ctr=det[g_ctr[j], 4].item() if g_ctr[j] >= 0 else 0.0,
                                best_iou=float(best)))
    dets = pd.DataFrame(det_rows, columns=['stem', 'x1', 'y1', 'x2', 'y2', 'conf', 'tp_iou', 'tp_ctr'])
    gts = pd.DataFrame(gt_rows)
    tag = f'{run.replace("/", "__")}_{variant}_{sp}'
    dets.to_csv(f'{OUT}/dets_{tag}.csv', index=False); gts.to_csv(f'{OUT}/gts_{tag}.csv', index=False)

    rows = []
    for scope, keep in [('all', None)] + [(m, m) for m in ['1호기', '2호기', '3호기']]:
        d = dets if keep is None else dets[dets.stem.map(split.machine) == keep]
        n = len(gts) if keep is None else (gts.stem.map(split.machine) == keep).sum()
        r_iou, r_ctr = summarize(d, n, 'tp_iou'), summarize(d, n, 'tp_ctr')
        rows.append(dict(run=run, imgsz=imgsz, input=variant, split=sp, scope=scope, n_img=len(stems) if keep is None else
                         (split.loc[stems].machine == keep).sum(), n_gt=n,
                         **r_iou, **{f'ctr_{k}': v for k, v in r_ctr.items()}))
    return rows


if __name__ == '__main__':
    # 사용: evaluate.py                                   → MODELS 전체, 입력 3종, eval/out
    #       evaluate.py --runs v2/B_s0 v2/B_s1 --inputs clean --out eval/out_v2 [--size 416]
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='*'); ap.add_argument('--inputs', nargs='*', default=['clean', 'marked', 'masked'])
    ap.add_argument('--out'); ap.add_argument('--size', type=int, default=416)
    args = ap.parse_args()
    if args.out:
        OUT = args.out if os.path.isabs(args.out) else f'{K}/{args.out}'
        os.makedirs(OUT, exist_ok=True)
    targets = [(r, f'{K}/runs/{r}/weights/best.pt', args.size) for r in args.runs] if args.runs else MODELS
    all_rows = []
    for run, w, imgsz in targets:
        if not os.path.exists(w):
            print('skip (없음):', w); continue
        model = load_model(w)
        for variant in args.inputs:   # 입력 전처리: clean=인페인팅 / marked=원본(마커) / masked=마스킹
            for sp in ['val', 'test']:
                rows = evaluate(run, w, imgsz, variant, sp, model)
                all_rows += rows
                r = rows[0]
                print(f'{run:14s} {variant:6s} {sp:4s}  AP50={r["AP50"]:.4f}  P@.25={r["P_at25"]:.3f}  R@.25={r["R_at25"]:.3f}'
                      f'  F1*={r["bestF1"]:.3f}@{r["bestF1_conf"]:.2f}  ctrAP={r["ctr_AP50"]:.4f}  ctrR@.25={r["ctr_R_at25"]:.3f}')
        del model; torch.cuda.empty_cache()
    df = pd.DataFrame(all_rows)
    # 기존 결과에 run 단위로 갱신 (다른 run 결과는 유지)
    for fname, part in [('metrics.csv', df[df.scope == 'all'].drop(columns='scope')), ('per_machine.csv', df)]:
        f = f'{OUT}/{fname}'
        if args.runs and os.path.exists(f):
            # (run, input, split[, scope]) 단위로 교체: 같은 모델의 다른 입력 결과는 유지
            prev = pd.read_csv(f)
            keys = [c for c in ['run', 'input', 'split', 'scope'] if c in part.columns and c in prev.columns]
            new_keys = set(map(tuple, part[keys].astype(str).values))
            prev = prev[[tuple(x) not in new_keys for x in prev[keys].astype(str).values]]
            part = pd.concat([prev, part])
        part.to_csv(f, index=False)
