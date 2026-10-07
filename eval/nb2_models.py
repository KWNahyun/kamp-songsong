"""새 베이스라인(YOLOv8s, D-FINE-S) 추론 어댑터: eval/evaluate.py 의 load_model / predict_img 와 같은 형태
- load_model('yolov8:<best.pt>') / load_model('dfine:<config.yml>|<best.pth>')
- predict_img(model, img0, imgsz) → ((H, W), tensor N×5 [x1, y1, x2, y2, conf], 원본 픽셀)
- patch(E): 기존 스크립트가 쓰는 E.load_model / E.predict_img 를 이 함수로 바꿔 끼움 (YOLOv3 경로는 그대로 통과)
추론 설정: 신뢰도 ≥ 0.001 (E.CONF_MIN), YOLOv8 NMS IoU 0.5 (YOLOv3 평가와 같음), D-FINE 은 NMS 없음(상위 300 쿼리)
"""
import sys
import numpy as np
import torch
import cv2

K = '/data/knhyun/KAMP'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CONF_MIN, NMS_IOU = 0.001, 0.5


class _V8:
    def __init__(self, w):
        # ultralytics 는 device 값으로 CUDA_VISIBLE_DEVICES 를 덮어씀 → CUDA 를 먼저 초기화해 두면 덮어써도 영향 없음
        # (실행 시 CUDA_VISIBLE_DEVICES=<물리 GPU> 로 고르고, 여기서는 보이는 0번 사용)
        torch.zeros(1, device=DEVICE)
        from ultralytics import YOLO
        self.m = YOLO(w)

    @torch.no_grad()
    def __call__(self, img0, imgsz):
        r = self.m.predict(img0, imgsz=imgsz, conf=CONF_MIN, iou=NMS_IOU, max_det=300, device=DEVICE, verbose=False)[0]
        b = r.boxes
        return torch.cat([b.xyxy.cpu(), b.conf.cpu()[:, None]], 1) if len(b) else torch.zeros((0, 5))


class _DFINE:
    def __init__(self, spec):
        cfg_path, w = spec.split('|')
        sys.path.insert(0, f'{K}/third_party/D-FINE')
        from src.core import YAMLConfig
        cfg = YAMLConfig(cfg_path)
        ck = torch.load(w, map_location='cpu')
        cfg.model.load_state_dict(ck['ema']['module'] if 'ema' in ck else ck['model'])
        self.model = cfg.model.deploy().to(DEVICE).eval()
        self.post = cfg.postprocessor.deploy().to(DEVICE).eval()

    @torch.no_grad()
    def __call__(self, img0, imgsz):
        h, w = img0.shape[:2]
        rgb = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        # 학습 때와 같은 변환: PIL 이중선형 Resize(imgsz × imgsz) → [0, 1]
        from PIL import Image
        im = np.asarray(Image.fromarray(rgb).resize((imgsz, imgsz), Image.BILINEAR), dtype=np.float32) / 255.0
        x = torch.from_numpy(im.transpose(2, 0, 1).copy())[None].to(DEVICE)
        labels, boxes, scores = self.post(self.model(x), torch.tensor([[w, h]], device=DEVICE))
        keep = scores[0] >= CONF_MIN
        return torch.cat([boxes[0][keep], scores[0][keep][:, None]], 1).float().cpu()


def load_model(spec):
    kind, path = spec.split(':', 1)
    return _V8(path) if kind == 'yolov8' else _DFINE(path)


def predict_img(model, img0, imgsz):
    if img0.ndim == 2:
        img0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
    return img0.shape[:2], model(img0, imgsz)


def patch(E):
    old_load, old_pred = E.load_model, E.predict_img
    E.load_model = lambda s: load_model(s) if s.startswith(('yolov8:', 'dfine:')) else old_load(s)
    E.predict_img = lambda m, im, sz: predict_img(m, im, sz) if isinstance(m, (_V8, _DFINE)) else old_pred(m, im, sz)
    return E


def specs(model, seeds=(0, 1, 2)):
    """학습된 새 베이스라인 경로 {이름: spec}. model = yolov8s | dfine_s, 뒤에 변형 이름을 붙일 수 있음
    (예: yolov8s_scratch, dfine_s_pidray = 사전학습 조건 ①·③, yolov8s_tpB = 옮겨 심기 학습). runs/nb2/<model>_s<seed>"""
    import os
    if model.startswith('yolov8s'):
        return {f'nb2/{model}_s{s}': f'yolov8:{K}/runs/nb2/{model}_s{s}/weights/best.pt' for s in seeds}
    # D-FINE: 2단계(증강 정지 후)에서 전체 최고를 넘었을 때만 best_stg2 가 생김 → 없으면 best_stg1 이 전체 최고. 추론 구조는 변형과 무관하게 같은 설정
    pick = lambda d: f'{d}/best_stg2.pth' if os.path.exists(f'{d}/best_stg2.pth') else f'{d}/best_stg1.pth'
    return {f'nb2/{model}_s{s}': f'dfine:{K}/configs_nb2/dfine_s_kamp.yml|{pick(f"{K}/runs/nb2/{model}_s{s}")}' for s in seeds}
