"""Sequential GPU jobs with durable status; failed jobs do not masquerade as done."""
from pathlib import Path
import os,sys,json,subprocess,time,datetime,hashlib
ROOT=Path('/home/viplab/contest'); EXP=ROOT/'experiments/kamp_pilot_v1'
PY=str(ROOT/'.detector-venv/bin/python')
env=os.environ.copy()
env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1',TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',YOLO_CONFIG_DIR=str(EXP/'ultralytics_settings'))
(EXP/'ultralytics_settings').mkdir(exist_ok=True)

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
    if (EXP/'queue_status.json').exists():
        raise RuntimeError('This experiment already has a run record. Use a new version directory; do not overwrite existing trials.')
    jobs=[('yolov8s',[PY,str(EXP/'train_yolo.py')]),
          ('dfine_s',[PY,str(EXP/'train_dfine.py'),'-c',str(EXP/'dfine_s.yml'),'-t',str(EXP/'dfine_s_init.pth'),'--device','cuda:0','--seed','20260929']),
          ('yolov8s_p2',[PY,str(EXP/'train_yolo.py'),'--p2']),
          ('dfine_s_p2',[PY,str(EXP/'train_dfine.py'),'--p2','-c',str(EXP/'dfine_s_p2.yml'),'-t',str(EXP/'dfine_s_p2_init.pth'),'--device','cuda:0','--seed','20260929'])]
    state={'started':now(),'pid':os.getpid(),'test_used':False,'jobs':[]}
    def save():
        temp=EXP/'queue_status.tmp';temp.write_text(json.dumps(state,indent=2));temp.replace(EXP/'queue_status.json')
    for name,cmd in jobs:
        item={'name':name,'status':'running','started':now(),'command':cmd};state['jobs'].append(item);save()
        with (EXP/f'{name}.log').open('w') as log:
            proc=subprocess.Popen(cmd,cwd=EXP,env=env,stdout=log,stderr=subprocess.STDOUT)
            item['pid']=proc.pid;save();code=proc.wait()
        item.update(status='trained' if code==0 else 'failed',returncode=code,finished=now());save()
        if code==0:
            evaluation=[PY,str(EXP/'evaluate_common.py'),'--model',name]
            with (EXP/f'{name}_evaluation.log').open('w') as log:
                code=subprocess.call(evaluation,cwd=EXP,env=env,stdout=log,stderr=subprocess.STDOUT)
            item['evaluation_returncode']=code
            item['status']='complete' if code==0 else 'evaluation_failed';save()
    state['finished']=now();save()

if __name__=='__main__':main()
