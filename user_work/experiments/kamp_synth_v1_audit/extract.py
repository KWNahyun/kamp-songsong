from pathlib import Path
import zipfile,hashlib,json,stat
r=Path('/home/viplab/contest/data/synthetic');p=r/'incoming/kamp_synth_test_v1.zip.part'
assert p.stat().st_size==506422446
z=zipfile.ZipFile(p);infos=z.infolist();print('members',len(infos),'expanded',sum(x.file_size for x in infos),flush=True)
for i in infos:
 dest=(r/i.filename).resolve();assert dest.is_relative_to(r.resolve());assert not stat.S_ISLNK(i.external_attr>>16)
assert z.testzip() is None
assert not (r/'kamp_synth_test_v1').exists()
z.extractall(r);z.close();target=p.with_suffix('');p.rename(target)
j={'source_url':'https://drive.google.com/file/d/1pw7gny3QzQcro2UKxw5xZ9TtBfroC7mK/view','download_date_KST':'2026-10-06','archive':str(target),'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'zip_crc_verified':True,'members':len(infos),'uncompressed_bytes':sum(x.file_size for x in infos),'extracted_root':str(r/'kamp_synth_test_v1')}
(r/'download_manifest.json').write_text(json.dumps(j,indent=2));print(json.dumps(j,indent=2))
