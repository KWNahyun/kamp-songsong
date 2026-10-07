"""One GPU job at a time, persistent state, exclusive lock, no result analysis."""
import os,json,time,subprocess,datetime,fcntl,hashlib,math,csv
from pathlib import Path
E=Path(__file__).parent

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
 lock=(E/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert json.loads((E/'preflight.json').read_text())['passed']
 manifest=json.loads((E/'run_manifest.json').read_text())
 for f,digest in manifest['sha256'].items():
    assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==digest, 'Changed experiment input: '+f
 path=E/'queue_status.json'
 if path.exists():raise RuntimeError('Existing queue state: do not overwrite or duplicate trials')
 state={'pid':os.getpid(),'started':now(),'status':'running','test_used':False,'analysis_performed':False,'jobs':manifest['jobs']}
 def save():
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(state,ensure_ascii=False,indent=2));tmp.replace(path)
 env=os.environ.copy();env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1',TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1')
 save()
 for job in state['jobs']:
    # Leave headroom; never alter another GPU user's processes or experimental batch size.
    while True:
        try:
            free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits','-i','0'],text=True).strip())
        except Exception:free=0
        if free>=18000:break
        job.update(status='waiting_for_vram',free_vram_MiB=free);save();time.sleep(30)
    job.update(status='running',started=now());save()
    with (E/'logs'/f"{job['name']}.log").open('x') as log:
        proc=subprocess.Popen(job['command'],cwd=E,env=env,stdout=log,stderr=subprocess.STDOUT)
        job['pid']=proc.pid;save();code=proc.wait()
    job.update(returncode=code,finished=now())
    run=E/'runs'/job['name']
    try:
        if code:raise RuntimeError(f'Process exit {code}')
        if job['family']=='dfine':
            rows=[json.loads(x) for x in (run/'log.txt').read_text().splitlines() if x.strip()]
            assert [x['epoch'] for x in rows]==list(range(30)),'Missing epochs'
            assert (run/'last.pth').exists() and (run/'best_stg1.pth').exists(),'Missing checkpoint'
            assert all(math.isfinite(v) for r in rows for k,v in r.items() if k.startswith('train_') and isinstance(v,(float,int)))
        else:
            with (run/'results.csv').open() as f:rows=list(csv.DictReader(f))
            assert len(rows)==30,'Missing epochs'
            assert (run/'weights/best.pt').exists() and (run/'weights/last.pt').exists(),'Missing checkpoint'
        job['status']='trained'
    except Exception as ex:
        job.update(status='failed',error=str(ex))
    save()
 state.update(status='finished' if all(x['status']=='trained' for x in state['jobs']) else 'finished_with_failures',finished=now());save()
if __name__=='__main__':main()
