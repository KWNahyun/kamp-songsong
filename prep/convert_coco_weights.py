"""외부 사전학습 가중치 변환 (결과보고서 '외부 데이터 활용' 항목 참고)
입력 : weights/yolov3-spp.weights  (https://pjreddie.com/media/files/yolov3-spp.weights, MS COCO 80 클래스)
출력 : weights/yolov3-spp-coco.pt  (train.py --weights 로 사용. 1 클래스 모델에는 형상이 같은 텐서만 전이됨)
"""
import hashlib
import sys
import torch

K = '/data/knhyun/KAMP'
sys.path.insert(0, f'{K}/yolov3')
from models import Darknet, load_darknet_weights

SRC = f'{K}/weights/yolov3-spp.weights'
SRC_MD5 = '6c569c9ef748131c2d96b2deab446a5f'
CFG80 = f'{K}/weights/yolov3-spp-coco80.cfg'
OUT = f'{K}/weights/yolov3-spp-coco.pt'

md5 = hashlib.md5(open(SRC, 'rb').read()).hexdigest()
assert md5 == SRC_MD5, f'md5 불일치: {md5}'

# 과제 cfg(1 클래스)의 헤드를 COCO 80 클래스로 바꾼 cfg 생성
cfg = open(f'{K}/yolov3/yolov3-spp.cfg').read().splitlines()
cfg = ['filters=255' if l == 'filters=18' else 'classes=80' if l == 'classes=1' else l for l in cfg]
open(CFG80, 'w').write('\n'.join(cfg) + '\n')

model = Darknet(CFG80)
load_darknet_weights(model, SRC)
sd = model.state_dict()
torch.save({'epoch': -1, 'best_fitness': 0.0, 'training_results': None, 'model': sd, 'optimizer': None}, OUT)

target = Darknet(f'{K}/yolov3/yolov3-spp.cfg').state_dict()
n_ok = sum(1 for k, v in sd.items() if k in target and target[k].numel() == v.numel())
print(f'saved {OUT} | 전이 가능 텐서 {n_ok}/{len(sd)}')
