"""Read complete entries before large weights from a still-downloading ZIP.
Full archive extraction and CRC are subsequently handled by acquire.py.
"""
from pathlib import Path
import struct,zlib,binascii
ROOT=Path(__file__).resolve().parents[1];target=(ROOT/'data/raw').resolve();count=0
with (ROOT/'data/04_xray_original.zip').open('rb') as f:
 while True:
  head=f.read(30)
  if head[:4]!=b'PK\x03\x04':break
  _,version,flags,method,tm,dt,crc,compressed,size,nlen,elen=struct.unpack('<4s5H3I2H',head)
  name=f.read(nlen).decode('utf8');extra=f.read(elen)
  if '/실습별 가중치파일/' in name:break
  dest=(target/name).resolve()
  if not dest.is_relative_to(target):raise ValueError('unsafe path')
  if name.endswith('/'):dest.mkdir(parents=True,exist_ok=True);continue
  if method==8:
   dec=zlib.decompressobj(-15);parts=[]
   while not dec.eof:
    chunk=f.read(2**20)
    if not chunk:raise ValueError('incomplete member')
    parts.append(dec.decompress(chunk))
   f.seek(-len(dec.unused_data),1);content=b''.join(parts)
  elif method==0 and not flags&8:content=f.read(compressed)
  else:raise ValueError((method,flags,name))
  if flags&8:
   d=f.read(4)
   if d==b'PK\x07\x08':crc,compressed,size=struct.unpack('<III',f.read(12))
   else:crc,compressed,size=struct.unpack('<III',d+f.read(8))
  if len(content)!=size or binascii.crc32(content)&0xffffffff!=crc:raise ValueError('CRC '+name)
  dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(content);count+=1
print('verified complete prefix files',count)
