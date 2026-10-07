"""Live current-job log; closing this viewer does not stop training."""
from pathlib import Path
import time,json
E=Path(__file__).parent;active=None;offset=0
try:
 while True:
  s=json.loads((E/'queue_status.json').read_text());name=s.get('active')
  if name!=active:
   active=name;offset=0;print('\nCompleted',sum(j['status']=='completed' for j in s['jobs']),'/',len(s['jobs']),'Active:',name,flush=True)
  if active:
   p=E/'logs'/f'{active}.log'
   if p.exists():
    with p.open(errors='replace') as f:f.seek(offset);chunk=f.read();offset=f.tell()
    if chunk:print(chunk,end='',flush=True)
  if s['status'] in ['completed','completed_with_failures','stopped_repeated_failures']:print(s['status']);break
  time.sleep(2)
except KeyboardInterrupt:print('\nViewer closed; training continues.')
