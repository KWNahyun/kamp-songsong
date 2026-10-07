"""Official public attachments; retain source files and SHA256 metadata."""
from pathlib import Path
import requests, hashlib, json, zipfile
ROOT=Path(__file__).resolve().parents[1]
FILES=[('8075','sources/task_statement.hwpx'),('8076','sources/report_template.hwpx'),('8080','data/04_xray_original.zip')]
def main():
    session=requests.Session(); session.headers['User-Agent']='Mozilla/5.0'
    meta=[]
    for seq,rel in FILES:
        p=ROOT/rel;p.parent.mkdir(parents=True,exist_ok=True)
        if not p.exists() or not zipfile.is_zipfile(p):
            tmp=p.with_suffix(p.suffix+'.part')
            with session.post('https://www.kamp-ai.kr/cptFileDownload',data={'fileSeq':seq},stream=True,timeout=120) as r:
                r.raise_for_status()
                with tmp.open('wb') as f:
                    for chunk in r.iter_content(2**20):f.write(chunk)
            if not zipfile.is_zipfile(tmp):raise ValueError(f'Not a valid archive: {tmp}')
            tmp.replace(p)
        h=hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda:f.read(2**20),b''):h.update(b)
        meta.append(dict(file=rel,fileSeq=seq,bytes=p.stat().st_size,sha256=h.hexdigest(),source_page='https://www.kamp-ai.kr/contestNoticeDetail?CPT_NOTICE_SEQ=28',download_endpoint='https://www.kamp-ai.kr/cptFileDownload',method='POST'))
        print(rel,meta[-1]['bytes'],flush=True)
    (ROOT/'sources/download_manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
    z=zipfile.ZipFile(ROOT/'data/04_xray_original.zip')
    target=(ROOT/'data/raw').resolve();target.mkdir(parents=True,exist_ok=True)
    for info in z.infolist():
        dest=(target/info.filename).resolve()
        if not dest.is_relative_to(target):raise ValueError('Unsafe zip member')
        if info.is_dir():dest.mkdir(parents=True,exist_ok=True);continue
        dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists() and dest.stat().st_size==info.file_size:continue
        with z.open(info) as src,dest.open('wb') as dst:
            for b in iter(lambda:src.read(2**20),b''):dst.write(b)
    bad=z.testzip()
    if bad:raise ValueError(f'CRC failure: {bad}')
    print('ZIP CRC verified; extracted',len(z.infolist()),'members',flush=True)
if __name__=='__main__':main()
