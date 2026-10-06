"""분할 검증: 내가 쓰는 train/val/test 목록이 정해진 분할과 같은지 확인합니다.

사용:
    python tools/verify_split.py --train my_train.txt --val my_val.txt --test my_test.txt [--root .]
  목록 파일: 한 줄에 이미지 하나 (경로 또는 파일 이름, 확장자 무관)
검사:
  1) 각 split 의 이미지 집합이 split_manifest.csv 와 정확히 같은가
  2) 같은 (장비, 날짜) 묶음이 두 split 에 걸쳐 있지 않은가
  3) 중복 이미지가 없는가
"""
import argparse, os
import pandas as pd


def ids(path):
    return [os.path.splitext(os.path.basename(l.strip()))[0] for l in open(path) if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    for s in ['train', 'val', 'test']:
        ap.add_argument(f'--{s}', required=True)
    ap.add_argument('--root', default='.')
    a = ap.parse_args()
    man = pd.read_csv(f'{a.root}/split_manifest.csv').set_index('image_id')
    ok = True
    mine = {s: ids(getattr(a, s)) for s in ['train', 'val', 'test']}
    for s, lst in mine.items():
        want = set(man.index[man.split == s]); got = set(lst)
        if len(lst) != len(got):
            ok = False; print(f'[{s}] 중복 {len(lst) - len(got)}개')
        miss, extra = want - got, got - want
        if miss or extra:
            ok = False; print(f'[{s}] 빠진 이미지 {len(miss)}개, 다른 split 이미지 {len(extra)}개 (예: {sorted(extra)[:3]})')
        else:
            print(f'[{s}] {len(got)}장 일치')
    rows = [(i, s) for s, lst in mine.items() for i in lst if i in man.index]
    g = pd.DataFrame(rows, columns=['image_id', 'split']).assign(group=lambda d: d.image_id.map(man.group))
    leak = g.groupby('group').split.nunique()
    if (leak > 1).any():
        ok = False; print('같은 (장비, 날짜) 묶음이 여러 split 에 있음:', leak[leak > 1].index.tolist()[:5])
    print('결과:', '통과' if ok else '실패')


if __name__ == '__main__':
    main()
