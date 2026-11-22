"""Finish reports only after the sealed grid and original analysis complete."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from pathlib import Path
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from common import read,write,sha

def main():
    run=ROOT/'results/20261020_phase3b_v3_run01'
    extension=run/'execution_extensions/finalization_driver_v1'
    extension.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,extension/'finalize_when_ready.py')
    write(extension/'manifest.json',{'source_sha256':sha(__file__),'no_training':True,
        'trigger':'execution completed AND original analysis_summary.complete true',
        'target_sha256':sha(ROOT/'scripts/finalize_analysis.py')})
    deadline=time.monotonic()+21600
    while time.monotonic()<deadline:
        failure=list((ROOT/'results/failures').glob('*.json'))
        failure+=list((run/'execution_extensions').glob('*/failure.json'))
        if failure:raise RuntimeError(f'Execution failure evidence found: {failure}')
        if (run/'analysis_summary.json').exists() and read(run/'analysis_summary.json').get('complete'):
            assert read(run/'execution_status.json')['stage']=='completed'
            assert sha(ROOT/'scripts/finalize_analysis.py')==read(extension/'manifest.json')['target_sha256']
            result=subprocess.run([sys.executable,str(ROOT/'scripts/finalize_analysis.py')])
            if result.returncode:raise RuntimeError(f'Final analysis exited {result.returncode}')
            write(extension/'completion.json',{'final_analysis_completed':True})
            print('Final analysis and preservation audit completed',flush=True)
            return
        time.sleep(10)
    raise TimeoutError('Finalization wait exceeded six hours')

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        import traceback
        run=ROOT/'results/20261020_phase3b_v3_run01'
        write(run/'execution_extensions/finalization_driver_v1/failure.json',{'exception':repr(exc),'traceback':traceback.format_exc()})
        raise
