"""Compact live progress for all active runs; Ctrl-C stops only this viewer."""
from pathlib import Path
import json,time,datetime
E=Path(__file__).parent;previous=None
try:
 while True:
  s=json.loads((E/'queue_status.json').read_text());rows=[]
  for active in s.get('active_jobs',[]):
   p=E/'runs'/active['name']/'log.txt';history=[]
   if p.exists():
    for line in p.read_text().splitlines():
     try:history.append(json.loads(line))
     except json.JSONDecodeError:pass
   last=history[-1] if history else {};rows.append((active['name'],active['phase'],len(history),last.get('train_loss')))
  key=(s['status'],sum(j['status']=='completed' for j in s['jobs']),rows)
  if key!=previous:
   print('\n',datetime.datetime.now().strftime('%H:%M:%S'),'|',s['status'],'| completed',key[1],'/',len(s['jobs']),flush=True)
   for name,phase,epochs,loss in rows:print(name,'|',phase,'| epochs',str(epochs)+'/30','| loss',f'{loss:.4f}' if loss is not None else '-',flush=True)
   previous=key
  if s['status'] in ['completed','completed_with_failures','stopped_repeated_failures']:break
  time.sleep(3)
except KeyboardInterrupt:print('\nViewer closed. Training continues.')
