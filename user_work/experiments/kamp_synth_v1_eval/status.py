from pathlib import Path
import json,datetime
E=Path(__file__).parent;P=json.loads((E/'protocol.json').read_text());done=0
for j in P['jobs']:
 d=E/'runs'/j['name'];stage='대기'
 if (d/'analysis/complete.json').exists():stage='추론·분석 완료';done+=1
 elif (d/'complete.json').exists():stage='추론 완료 / 분석 중'
 elif (d/'status.json').exists():
  s=json.loads((d/'status.json').read_text());stage=f"{s['processed']:,}/{s['total']:,}장 · {s['images_per_second']:.1f}장/초"
 print(f"{j['name']:38} {stage}")
print(f'완료 {done}/{len(P["jobs"])}개 실험')
if (E/'completion.json').exists():print((E/'completion.json').read_text())
