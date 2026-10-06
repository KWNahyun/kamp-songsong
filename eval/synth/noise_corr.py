"""실제 잡음 vs 추가한 흰 잡음의 공간 상관: 2×2 평균 후 잡음 / 화소 잡음 비 (흰 잡음이면 0.5)
결함은 2×2 코어이므로, 이 비가 다르면 같은 화소 CNR 이라도 결함 크기에서의 실효 CNR 이 다르다
출력: eval/synth/noise_corr.csv, 콘솔
"""
import numpy as np, pandas as pd, cv2
K = '/data/knhyun/KAMP'
split = pd.read_csv(f'{K}/data/splits/split.csv'); vt = split[split.split.isin(['val', 'test'])]
mcol = 'machine' if 'machine' in vt.columns else None
rng = np.random.default_rng(0); rows = []
mad = lambda a, m: float(np.median(np.abs(a[m] - np.median(a[m]))) * 1.4826)   # 포화·평탄 영역 제외(m)
for r in vt.itertuples():
    im = cv2.imread(f'{K}/data/synth_val/erased/images/{r.stem}.png', 0).astype(np.float32)
    for lab, add in [('real', 0), ('real+white6', 6)]:
        x = im + rng.normal(0, add, im.shape) if add else im
        res = x - cv2.medianBlur(np.clip(x, 0, 255).astype(np.uint8), 5).astype(np.float32)
        hp = x - cv2.GaussianBlur(x, (0, 0), 3)           # 저주파 제거 후 잡음
        b2 = cv2.blur(hp, (2, 2))
        m = cv2.erode(((im > 10) & (im < 245)).astype(np.uint8), np.ones((9, 9))) > 0
        if m.sum() < 1000: continue
        rows.append(dict(stem=r.stem, machine=getattr(r, 'machine', None), kind=lab, s1=mad(hp, m), s2=mad(b2, m), ratio=mad(b2, m) / max(mad(hp, m), 1e-6), s_med=mad(res, m)))
D = pd.DataFrame(rows); D.to_csv(f'{K}/eval/synth/noise_corr.csv', index=False)
print(D.groupby(['machine', 'kind'])[['s1', 's2', 'ratio', 's_med']].median().round(3))
w = np.random.default_rng(1).normal(0, 6, (512, 512)).astype(np.float32); hp = w - cv2.GaussianBlur(w, (0, 0), 3)
a = np.ones_like(hp, bool); print('순수 흰 잡음 ratio', round(mad(cv2.blur(hp, (2, 2)), a) / mad(hp, a), 3))
