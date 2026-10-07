"""Fetch only Dataset #1 / 40 kV members, verify ZIP CRC, retain original TIFFs."""
from pathlib import Path
import json,urllib.request,concurrent.futures,struct,zlib,time,hashlib
R=Path(__file__).resolve().parents[1];S=R/'source';U='https://zenodo.org/api/records/10579608/files/Submission.zip/content'
idx=json.loads((S/'zip_index.json').read_text());items=[x for x in idx if 'Dataset #1/40kV_40W_100ms_10avg/' in x['name'] and x['size']]
start=min(x['offset'] for x in items);end=max(x['offset']+30+len(x['name'].encode())+65535+x['compressed'] for x in items)
chunk=8*1024*1024;parts=list(range(start,end,chunk));cache=S/'ranges';cache.mkdir(exist_ok=True)
def fetch(a):
 b=min(a+chunk,end)-1;p=cache/str(a)
 if p.exists() and p.stat().st_size==b-a+1:return
 for attempt in range(5):
  try:
   req=urllib.request.Request(U,headers={'Range':f'bytes={a}-{b}'})
   with urllib.request.urlopen(req,timeout=120) as f:
    assert f.status==206
    data=f.read();assert len(data)==b-a+1
   p.write_bytes(data);return
  except Exception:
   if attempt==4:raise
   time.sleep(2**attempt)
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
 for k,_ in enumerate(ex.map(fetch,parts),1):print(f'ranges {k}/{len(parts)}',flush=True)
def read_at(a,n):
 out=bytearray()
 while n:
  base=start+((a-start)//chunk)*chunk;take=min(n,base+chunk-a)
  with (cache/str(base)).open('rb') as f:f.seek(a-base);out.extend(f.read(take))
  n-=take;a+=take
 return bytes(out)
manifest=[]
for x in items:
 h=struct.unpack('<IHHHHHIIIHH',read_at(x['offset'],30));assert h[0]==0x04034b50
 data=read_at(x['offset']+30+h[-2]+h[-1],x['compressed'])
 raw=zlib.decompress(data,-15) if h[3]==8 else data
 assert len(raw)==x['size'] and zlib.crc32(raw)==x['crc']
 p=S/x['name'];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
 manifest.append({**x,'sha256':hashlib.sha256(raw).hexdigest()})
(S/'download_manifest.json').write_text(json.dumps({'url':U,'subset':'Dataset #1/40kV_40W_100ms_10avg','archive_checksum_not_verified':'partial retrieval; each extracted member verified against ZIP CRC32','members':manifest},indent=2))
print('DONE',len(manifest),flush=True)
