from remotezip import RemoteZip
from pathlib import Path
import csv,io,json
root=Path(__file__).resolve().parents[1]
with RemoteZip('https://zenodo.org/api/records/10579608/files/Submission.zip/content') as z:
 infos=[{'name':i.filename,'offset':i.header_offset,'compressed':i.compress_size,'size':i.file_size,'crc':i.CRC} for i in z.infolist()]
 (root/'source/zip_index.json').write_text(json.dumps(infos,indent=2))
 for i in infos:
  n=i['name']
  if 'Dataset #1/40kV' in n and n.endswith('.csv'):
   b=z.read(n);p=root/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
   rows=list(csv.DictReader(io.StringIO(b.decode())))
   print(n,len(rows),'chickens',sorted(set(r['Chicken_ID'] for r in rows)),'bones',sorted(set(r['Bone_ID'] for r in rows)),flush=True)
