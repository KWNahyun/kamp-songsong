"""의사 라벨 방법 비교용 train 교차 검증 조각 (3-fold)
- 대상: GT train 350장. 묶음 = (장비, 날짜) → 같은 날 사진은 같은 조각
- 층 = 장비 × 시편 종류(결함 1개 / 3개). 층마다 날짜 묶음 배정 조합을 무작위 탐색해 조각 크기를 고르게 (seed 42)
- 조각 k: 교사 학습 = 나머지 두 조각, 의사 라벨 방법 평가 = 조각 k (공식 TXT 와 비교)
  교사의 최적 시점 선택·임계값 결정은 공식 val 사용 (평가 조각과 겹치지 않음). test 는 사용하지 않음
출력: data/splits/cv3.csv, data/splits/cv3_f<k>_{train,hold}.txt, data/cv3_f<k>.data
"""
import numpy as np
import pandas as pd

K = '/data/knhyun/KAMP'
DATA = f'{K}/data'
NF, SEED = 3, 42
s = pd.read_csv(f'{DATA}/splits/split.csv')
tr = s[s.split == 'train'].copy()
grp = tr.groupby('group').agg(machine=('machine', 'first'), kind=('kind', lambda k: k.mode()[0]), n=('stem', 'size')).reset_index()
rng = np.random.default_rng(SEED)
fold_of = {}
for (m, k), g in grp.groupby(['machine', 'kind']):
    groups, sizes = g.group.tolist(), g.n.values
    target = sizes.sum() / NF
    best, err_best = None, np.inf
    for _ in range(20000):
        a = rng.integers(0, NF, len(groups))
        if len(groups) >= NF and len(set(a)) < NF:
            continue
        err = sum((sizes[a == f].sum() - target) ** 2 for f in range(NF))
        if err < err_best:
            best, err_best = a, err
    fold_of.update(dict(zip(groups, best)))
tr['fold'] = tr.group.map(fold_of)
tr[['stem', 'machine', 'kind', 'date', 'group', 'session', 'fold']].to_csv(f'{DATA}/splits/cv3.csv', index=False)
for f in range(NF):
    trn = tr[tr.fold != f].stem; hold = tr[tr.fold == f].stem
    open(f'{DATA}/splits/cv3_f{f}_train.txt', 'w').write('\n'.join(f'{DATA}/clean/images/{x}.png' for x in trn) + '\n')
    open(f'{DATA}/splits/cv3_f{f}_hold.txt', 'w').write('\n'.join(f'{DATA}/clean/images/{x}.png' for x in hold) + '\n')
    open(f'{DATA}/cv3_f{f}.data', 'w').write(
        f'classes=1\ntrain={DATA}/splits/cv3_f{f}_train.txt\nvalid={DATA}/splits/clean_val.txt\nnames={DATA}/classes.names\n')
print(pd.crosstab([tr.machine, tr.kind], tr.fold, margins=True))
print('결함 수:', tr.groupby('fold').n_obj.sum().to_dict(), '| 묶음 수:', tr.groupby('fold').group.nunique().to_dict())
assert tr.groupby('group').fold.nunique().max() == 1
