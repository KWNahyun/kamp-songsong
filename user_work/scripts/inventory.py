from pathlib import Path
import json,collections,hashlib,csv
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];RAW=ROOT/'data/raw';OUT=ROOT/'analysis'
files=[p for p in RAW.rglob('*') if p.is_file()]
rows=[];groups=collections.Counter()
for p in files:
 rel=p.relative_to(RAW);groups[(str(rel.parent),p.suffix.lower())]+=1
 rows.append({'path':str(rel),'bytes':p.stat().st_size,'extension':p.suffix.lower()})
with (OUT/'file_inventory.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=['path','bytes','extension']);w.writeheader();w.writerows(rows)
print('ALL FILES',len(files),'BY EXTENSION',collections.Counter(p.suffix.lower() for p in files))
for (folder,ext),n in groups.items():
 if n>=10:print(n,ext,folder)
print('\nCONFIG/DOC FILES')
for p in files:
 if p.suffix.lower() in ['.pdf','.hwp','.hwpx','.ipynb','.yaml','.yml','.names','.data'] or p.name in ['class_list.txt','classes.txt','obj.names','README.md']:
  print(p.relative_to(RAW),p.stat().st_size)
  if p.stat().st_size<10000 and p.suffix.lower() in ['.yaml','.yml','.names','.data','.txt']:print(p.read_text(errors='replace'))
