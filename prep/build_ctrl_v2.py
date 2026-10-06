"""학습량 대조군 (미라벨 요인, 분할 v2)
학생(U1_p990_s<seed>)과 학습 목록 크기·epoch·학습률 스케줄을 똑같이 맞추되, 의사 라벨 이미지 대신
GT train 350장을 반복해서 채운다. → 학생 vs 대조군 차이 = '미라벨 이미지의 순수 효과'
출력: data/splits/ctrl_v2_s<seed>_train.txt, data/ctrl_v2_s<seed>.data
"""
import numpy as np
import pandas as pd

K = '/data/knhyun/KAMP'
DATA = f'{K}/data'
train = [f'{DATA}/clean/images/{s}.png' for s in pd.read_csv(f'{DATA}/splits/split.csv').query('split=="train"').stem]
for seed in [0, 1, 2]:
    n = sum(1 for _ in open(f'{DATA}/splits/pl_v2_s{seed}_p990_train.txt'))
    rng = np.random.default_rng(seed)
    reps, extra = divmod(n, len(train))
    lst = train * reps + list(rng.choice(train, extra, replace=False))
    open(f'{DATA}/splits/ctrl_v2_s{seed}_train.txt', 'w').write('\n'.join(lst) + '\n')
    open(f'{DATA}/ctrl_v2_s{seed}.data', 'w').write(
        f'classes=1\ntrain={DATA}/splits/ctrl_v2_s{seed}_train.txt\nvalid={DATA}/splits/clean_val.txt\nnames={DATA}/classes.names\n')
    print(f'seed {seed}: 학생 목록 {n}장 = 대조군 목록 {len(lst)}장 (GT {len(train)}장 × {reps} + {extra})')
