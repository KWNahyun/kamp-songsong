"""Resolve paths to existing 640-letterboxed training data; never reads test."""
from pathlib import Path
import argparse,json,yaml
p=argparse.ArgumentParser();p.add_argument('--data640',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
r=Path(__file__).resolve().parents[1];c=yaml.safe_load((r/'configs/train_mal.yml').read_text())
if a.output.exists():raise FileExistsError(a.output)
for split in ['train','val']:
 d=a.data640.resolve();ann=d/'annotations'/f'{split}.json';images=d/'images'/split
 j=json.loads(ann.read_text());assert j['images'] and j['annotations']
 assert all((images/x['file_name']).is_file() for x in j['images'])
 assert all(x['width']==640 and x['height']==640 for x in j['images'])
 c[split+'_dataloader']['dataset'].update(img_folder=str(images),ann_file=str(ann))
c['output_dir']=str(a.output.resolve().parent/'mal_run')
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(yaml.safe_dump(c,sort_keys=False))
print(a.output)
