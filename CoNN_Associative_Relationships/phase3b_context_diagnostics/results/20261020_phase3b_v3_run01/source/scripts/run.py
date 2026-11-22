"""Launch/resume isolated Phase3B; no writes to historical directories."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import argparse
from common import write,read

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['preflight','execute','verify','report']);p.add_argument('--run',default='20261020_phase3b_v3_run01')
    args=p.parse_args();run=ROOT/'results'/args.run
    if args.stage=='preflight':
        from preflight import run_preflight
        print(run_preflight(run),flush=True)
    elif args.stage=='execute':
        from training import execute
        execute(run)
    elif args.stage=='verify':
        from verify import verify_run
        print(verify_run(run),flush=True)
    elif args.stage=='report':
        from reporting import report
        report(run)

if __name__=='__main__':
    try: main()
    except BaseException as exc:
        import traceback
        directory=ROOT/'results'/'failures';directory.mkdir(parents=True,exist_ok=True)
        import time
        write(directory/f'{time.time_ns()}.json',{'exception':repr(exc),'traceback':traceback.format_exc(),'argv':sys.argv})
        raise
