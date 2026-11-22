"""Freeze new development scope and preserve prior evidence, without fitting."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path[:] = [p for p in sys.path if Path(p or '.').resolve() != ROOT/'scripts']
sys.path.insert(0,str(ROOT/'src'))

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''): h.update(b)
    return h.hexdigest()

def main():
    run=ROOT/'results/runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_phase23_v1')
    run.mkdir(parents=True,exist_ok=False)
    def write(n,v): (run/n).write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
    from conn_assoc.protocol import frozen_constants,protocol_sha256,manifest
    from conn_assoc.data import verify_archive,verify_prepared_manifest
    assert protocol_sha256()==manifest()['protocol_sha256']
    old=ROOT/'results/runs/20261019T024656620209Z_audit_recovery_v1/experiment2.after.json'
    previous=json.loads(old.read_text())
    changed=[p for p,r in previous.items() if digest(Path(p))!=r['sha256']]
    assert not changed,changed
    verify_archive();verify_prepared_manifest()
    write('experiment2.before.json',previous)
    protected=[p for p in ROOT.rglob('*') if p.is_file() and not any(s.startswith('.venv') for s in p.relative_to(ROOT).parts) and not p.is_relative_to(run) and '__pycache__' not in p.parts]
    write('prior_associative_manifest.json',{str(p.relative_to(ROOT)):digest(p) for p in protected})
    for name in ('README.md','PROTOCOL_v2.md','MANIFEST.json','config/phase01.yaml','reports/EXPERIMENT_AUDIT_AND_RESULTS.md','audit/inspection.json','PROTOCOL_PHASE23_v1.md'):
        p=ROOT/name;q=run/'prior_snapshot'/name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
    c=frozen_constants()
    config={'version':'phase23_v1','steps':3200,'eval_interval':160,'batch_size':128,'hidden_sizes':[64,64],
      'learning_rates':[.01,.05,.1],'seeds':[400,401,402,403,404],'folds':c['development_cv']['folds'],
      'conditions':{'P2_dense':'A','P2_gaussian':'D','P2_rowshuffle':'B','P2_coordperm':'C','P2_fixed08':'S08','P2_fixed25':'S25',
        'P3_dense_masked':'A','P3_gaussian_masked':'D','P3_dense_context':'A','P3_gaussian_context':'D'},
      'context':{'rank':2,'floor':.05,'u_init':-6.,'v_init':0.,'contexts':['both','accelerometer','gyroscope']},
      'official_test_evaluation':False,'maximum_grid_wall_seconds':21600}
    write('config.frozen.json',config)
    write('protocol_seal.json',{'sealed_utc':datetime.now(timezone.utc).isoformat(),'original_protocol_sha256':protocol_sha256(),
      'amendment_sha256':digest(ROOT/'PROTOCOL_PHASE23_v1.md'),'config_sha256':digest(run/'config.frozen.json'),
      'authorization':'User explicitly requested Phase 2/3 development implementation and training; no Phase 4/final test',
      'git_commit':None,'original_experiment2_files_verified':len(previous)})
    import torch
    write('cpu_environment.before.json',{'executable':sys.executable,'python':sys.version,'torch':torch.__version__,
      'cuda_available':torch.cuda.is_available(),'device_count':torch.cuda.device_count(),'cpu_count':os.cpu_count()})
    for name,args in [('nvidia_smi',['nvidia-smi']),('cpu_pip_freeze',[sys.executable,'-m','pip','freeze'])]:
        r=subprocess.run(args,capture_output=True,text=True)
        (run/(name+'.log')).write_text(r.stdout+r.stderr,encoding='utf-8')
    (ROOT/'config/phase23_active_run.txt').write_text(str(run)+'\n')
    print(run)

if __name__=='__main__':main()
