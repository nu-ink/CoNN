"""Additive execution-only driver for disjoint future seed blocks on CPU.

Does not change frozen model/trainer/config files. The original serial runner
retains seeds400/401 and skips these already-completed jobs on arrival. Use
only while its current seed is400; stop scheduling before seed401 begins.
"""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import concurrent.futures
import time
from common import read,write,sha
from models import CONDITIONS

def fit(task):
    from training import train_job
    run,condition,seed,fold,cfg,deadline=task
    return train_job(Path(run),'grid',condition,seed,fold,'cpu',cfg,deadline)

def main():
    run=ROOT/'results/20261020_phase3b_v3_run01'
    manifest=read(run/'execution_manifest.frozen.json');cfg=read(run/'config.frozen.json')
    status=read(run/'execution_status.json')
    assert status['stage']=='grid' and status['last'][1]==400,'Requires untouched future seed blocks'
    assert manifest['backend']=='cpu'
    target=run/'execution_extensions/parallel_tail_v1';target.mkdir(parents=True,exist_ok=False)
    import shutil
    shutil.copy2(__file__,target/'parallel_tail.py')
    write(target/'manifest.frozen.json',{'kind':'execution_only_extension','training_source_manifest_sha256':sha(run/'source_manifest.json'),
        'driver_sha256':sha(__file__),'workers':4,'backend':'cpu','seeds':[402,403,404],
        'no_model_optimizer_data_or_statistical_changes':True,'canonical_jobs_shared_with_original_runner':True,
        'original_runner_retains_seeds':[400,401],'schedule_rule':'future blocks only; queue stops before original seed401 begins',
        'parameter': 'all original source hashes remain unchanged'})
    tasks=[(str(run),c,s,f,cfg,time.monotonic()+cfg['maximum_wall_seconds']-status['elapsed_seconds'])
        for s in [404,403,402] for f in reversed(range(7)) for c in reversed(CONDITIONS) if c not in manifest['omitted_conditions']]
    completed=0;queued=0;pending={}
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
            while tasks or pending:
                status=read(run/'execution_status.json')
                can_queue=status['stage']=='grid' and status['last'][1]==400
                while tasks and len(pending)<4 and can_queue:
                    task=tasks.pop(0);p=run/'grid'/f'{task[1]}_seed{task[2]}_fold{task[3]}'/'completion.json'
                    if p.exists():continue
                    pending[pool.submit(fit,task)]=task;queued+=1
                if not pending:break
                done,_=concurrent.futures.wait(pending,timeout=30,return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    task=pending.pop(future);result=future.result();completed+=1
                    print('tail complete',task[1:4],completed,flush=True)
                write(target/'progress.json',{'completed':completed,'queued':queued,'remaining_not_queued':len(tasks),
                    'original_runner_status':status,'queue_open':can_queue})
        write(target/'completion.json',{'completed_fits':completed,'logical_updates':completed*3200,
            'unlaunched_fits_retained_for_serial_runner':len(tasks)})
    except BaseException as exc:
        import traceback
        write(target/'failure.json',{'exception':repr(exc),'traceback':traceback.format_exc()})
        raise

if __name__=='__main__': main()
