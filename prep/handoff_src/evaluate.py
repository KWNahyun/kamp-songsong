"""공통 평가 스크립트 (numpy·pandas 만 필요) — 베이스라인과 같은 지표를 계산합니다.

사용:
    python tools/evaluate.py --pred predictions.csv --split test [--root .]

predictions.csv 형식 (원본 이미지 픽셀 좌표, 한 줄 = 검출 하나):
    image_id,x1,y1,x2,y2,conf
    001_20200623_003516(8),361.0,190.0,372.0,201.0,0.583
  - image_id: split_manifest.csv 의 image_id (확장자 없음)
  - AP 계산을 위해 낮은 신뢰도 검출까지 모두 넣으세요 (권장: conf ≥ 0.001, NMS 후)

지표:
  AP50      : IoU ≥ 0.5 매칭, 101점 보간 AP
  P@0.25    : conf ≥ 0.25 검출 중 정답 비율
  R@0.25    : 정답 결함 중 conf ≥ 0.25 로 찾은 비율
  ctr_*     : IoU 대신 '박스 중심 거리 ≤ 8px' 로 매칭한 같은 지표
              (결함이 2×2px 점인데 정답 박스 크기가 5~21px 로 들쭉날쭉해, 위치 적중을 따로 봄)
  매칭: 이미지마다 conf 높은 검출부터, 아직 매칭 안 된 정답과 짝지음 (IoU / 중심거리 각각 독립)
"""
import argparse
import numpy as np
import pandas as pd

IOU_T, CENTER_T = 0.5, 8.0


def load_gt(root, man_row):
    W, H = man_row.width, man_row.height
    rows = [list(map(float, l.split())) for l in open(f'{root}/{man_row.label_path}') if l.strip()]
    if not rows:
        return np.zeros((0, 4))
    a = np.array(rows)[:, 1:5]
    return np.stack([(a[:, 0] - a[:, 2] / 2) * W, (a[:, 1] - a[:, 3] / 2) * H,
                     (a[:, 0] + a[:, 2] / 2) * W, (a[:, 1] + a[:, 3] / 2) * H], 1)


def box_iou(a, b):
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    lt = np.maximum(a[:, None, :2], b[None, :, :2]); rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    return inter / (area(a)[:, None] + area(b)[None] - inter)


def match(det, gt):
    n, m = len(det), len(gt)
    tp_iou, tp_ctr = np.zeros(n, bool), np.zeros(n, bool)
    if n and m:
        g_iou, g_ctr = np.full(m, -1), np.full(m, -1)
        iou = box_iou(det[:, :4], gt)
        dc = np.linalg.norm((det[:, None, :2] + det[:, None, 2:4]) / 2 - (gt[None, :, :2] + gt[None, :, 2:]) / 2, axis=2)
        for i in np.argsort(-det[:, 4], kind='stable'):
            j = np.argmax(np.where(g_iou < 0, iou[i], -1))
            if iou[i, j] >= IOU_T and g_iou[j] < 0:
                tp_iou[i] = True; g_iou[j] = i
            d = np.where(g_ctr < 0, dc[i], np.inf); j = np.argmin(d)
            if d[j] <= CENTER_T:
                tp_ctr[i] = True; g_ctr[j] = i
    return tp_iou, tp_ctr


def compute_ap(recall, precision):
    mrec = np.concatenate(([0.], recall, [min(recall[-1] + 1E-3, 1.)]))
    mpre = np.concatenate(([0.], precision, [0.]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    trapz = getattr(np, 'trapezoid', None) or np.trapz
    return float(trapz(np.interp(x, mrec, mpre), x))


def summarize(conf, tp, n_gt):
    o = np.argsort(-conf, kind='stable'); conf, tp = conf[o], tp[o].astype(float)
    if not len(tp):
        return dict(AP50=0.0, P_at25=0.0, R_at25=0.0)
    ctp = np.cumsum(tp); rec = ctp / max(n_gt, 1); prec = ctp / np.arange(1, len(tp) + 1)
    s = conf >= 0.25
    return dict(AP50=compute_ap(rec, prec), P_at25=float(tp[s].mean()) if s.any() else 0.0, R_at25=float(tp[s].sum() / max(n_gt, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True); ap.add_argument('--split', default='test'); ap.add_argument('--root', default='.')
    a = ap.parse_args()
    man = pd.read_csv(f'{a.root}/split_manifest.csv')
    man = man[man.split == a.split]
    pred = pd.read_csv(a.pred)
    unknown = set(pred.image_id) - set(man.image_id)
    if unknown:
        raise SystemExit(f'이 split 에 없는 image_id {len(unknown)}개 (예: {list(unknown)[:3]}) — split 을 확인하세요')
    confs, tps_iou, tps_ctr, n_gt = [], [], [], 0
    per = []
    for r in man.itertuples():
        gt = load_gt(a.root, r); n_gt += len(gt)
        d = pred[pred.image_id == r.image_id][['x1', 'y1', 'x2', 'y2', 'conf']].values.astype(float)
        ti, tc = match(d, gt)
        confs.append(d[:, 4] if len(d) else np.zeros(0)); tps_iou.append(ti); tps_ctr.append(tc)
        per.append((r.machine, len(gt), d[:, 4] if len(d) else np.zeros(0), ti, tc))
    conf = np.concatenate(confs); ti = np.concatenate(tps_iou); tc = np.concatenate(tps_ctr)
    res = summarize(conf, ti, n_gt); rc = summarize(conf, tc, n_gt)
    print(f'split={a.split}  이미지 {len(man)}  정답 결함 {n_gt}  검출 {len(conf)}')
    print(f"AP50={res['AP50']:.4f}  P@0.25={res['P_at25']:.4f}  R@0.25={res['R_at25']:.4f}  "
          f"ctr_AP50={rc['AP50']:.4f}  ctr_R@0.25={rc['R_at25']:.4f}")
    for mc in sorted(set(p[0] for p in per)):
        sub = [p for p in per if p[0] == mc]
        c = np.concatenate([p[2] for p in sub]); t = np.concatenate([p[3] for p in sub]); g = sum(p[1] for p in sub)
        s = summarize(c, t, g)
        print(f'  {mc}: 결함 {g}  AP50={s["AP50"]:.4f}  R@0.25={s["R_at25"]:.4f}')


if __name__ == '__main__':
    main()
