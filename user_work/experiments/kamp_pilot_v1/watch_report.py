"""Finite companion to this training queue; refresh report until queue exits."""
from pathlib import Path
import time,json,subprocess,sys,datetime
EXP=Path(__file__).resolve().parent

def main():
    started=time.monotonic()
    while time.monotonic()-started<4*3600:
        subprocess.run([sys.executable,str(EXP/'summarize.py')],check=True)
        state=json.loads((EXP/'queue_status.json').read_text())
        if 'finished' in state:
            (EXP/'report_status.json').write_text(json.dumps({'updated':datetime.datetime.now().isoformat(),'finished':True,'all_jobs_complete':all(j['status']=='complete' for j in state['jobs'])},indent=2))
            return
        time.sleep(20)
    (EXP/'report_status.json').write_text(json.dumps({'finished':False,'reason':'4-hour watcher limit; inspect queue logs'}))

if __name__=='__main__':main()
