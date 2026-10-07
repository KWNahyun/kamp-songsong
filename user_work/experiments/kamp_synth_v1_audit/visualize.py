from pathlib import Path
import pandas as pd,numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
plt.rcParams.update({'font.family':'Noto Sans CJK JP','pdf.fonttype':42})
r=Path('/home/viplab/contest/data/synthetic/kamp_synth_test_v1');out=Path('/home/viplab/contest/experiments/kamp_synth_v1_audit');i=pd.read_csv(r/'images.csv');o=pd.read_csv(r/'objects.csv')
source='002_20200714_083054(2)';d=i[(i.source_image==source)&(i.split=='test')]
if d.empty:source=i[i.split=='test'].source_image.iloc[0];d=i[(i.source_image==source)&(i.split=='test')]
sets=['erased','pos_canon_k1.0','pos_random_k1.0','gvxr_size_x0.5','gvxr_size_x1','gvxr_size_x5','gvxr_shape_wire20','gvxr_material_plastic_x1']
fig,axs=plt.subplots(2,4,figsize=(16,8))
for ax,name in zip(axs.flat,sets):
 row=d[d['set']==name].iloc[0];a=np.array(Image.open(r/'images/test'/(row.image_id+'.png')));ax.imshow(a,cmap='gray',vmin=0,vmax=255)
 for b in o[o.image_id==row.image_id].itertuples():
  ax.add_patch(Rectangle((b.x1,b.y1),b.x2-b.x1,b.y2-b.y1,fill=False,ec='#FF8C00',lw=.9));ax.plot(b.obj_cx,b.obj_cy,'+',color='#00c8ff',ms=4)
 ax.set_title(name,fontsize=10);ax.axis('off')
fig.suptitle('동일 원본의 합성 조건별 전체 영상과 라벨\n주황: TXT 박스 / 청록: 합성 물체 중심 | '+source,fontsize=14);fig.tight_layout();fig.savefig(out/'audit_full_images.png',dpi=150);plt.close(fig)
# aligned crops with identical grayscale range
sets=['gvxr_size_x0.5','gvxr_size_x1','gvxr_size_x5','gvxr_shape_wire20','gvxr_material_Al_x1','gvxr_material_plastic_x1']
fig,axs=plt.subplots(2,6,figsize=(15,6))
for j,name in enumerate(sets):
 row=d[d['set']==name].iloc[0];b=o[o.image_id==row.image_id].iloc[0];a=np.array(Image.open(r/'images/test'/(row.image_id+'.png')));cx,cy=round(b.obj_cx),round(b.obj_cy);x0,y0=cx-22,cy-22
 for k in [0,1]:
  ax=axs[k,j];ax.imshow(a[max(0,y0):cy+22,max(0,x0):cx+22],cmap='gray',vmin=0,vmax=255,interpolation='nearest');ax.axis('off')
  if k==1:
   ax.add_patch(Rectangle((b.x1-x0,b.y1-y0),b.x2-b.x1,b.y2-b.y1,fill=False,ec='#FF8C00',lw=1.3));ax.add_patch(Rectangle((b.ex1-x0,b.ey1-y0),b.ex2-b.ex1,b.ey2-b.ey1,fill=False,ec='#00BCD4',lw=1,ls='--'))
 axs[0,j].set_title(name.replace('gvxr_','')+'\n대비 %.2f'%b.contrast,fontsize=10)
fig.suptitle('원본 픽셀 주변 확대: 위 영상 / 아래 TXT 박스와 평가 영역\n모든 패널 동일 밝기 범위 0–255, 최근접 보간',fontsize=13);fig.tight_layout();fig.savefig(out/'audit_crops.png',dpi=180);plt.close(fig)
