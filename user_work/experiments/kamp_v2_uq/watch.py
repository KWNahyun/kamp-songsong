"""Follow current job across queue transitions: python3 watch.py."""
from pathlib import Path
import json,time
E=Path(__file__).parent
current=None; offset=0; previous=None
try:
 while True:
  state=json.loads((E/'queue_status.json').read_text())
  active=next((j for j in state['jobs'] if j['status']=='running'),None)
  summary=(state['status'],sum(j['status']=='trained' for j in state['jobs']))
  if summary!=previous:
   print(f"\nQueue: {summary[0]} | completed {summary[1]}/{len(state['jobs'])}",flush=True);previous=summary
  if active:
   if current!=active['name']:current=active['name'];offset=0;print('\nMODEL:',current,flush=True)
   p=E/'logs'/f'{current}.log'
   if p.exists():
    with p.open(errors='replace') as f:
     f.seek(offset);chunk=f.read();offset=f.tell()
    if chunk:print(chunk,end='',flush=True)
  if state['status'] in ['finished','stopped_on_failure','finished_with_failures']:break
  time.sleep(2)
except KeyboardInterrupt:
 print('\nViewer closed; background training continues.')
