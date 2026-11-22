"""Create initial frozen configurations and inventory original files, once only."""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from conn_exp04 import ROOT
from conn_exp04.core import Config, digest_json

base=ROOT/'experiments'/'experiment_04'
config=base/'config'
config.mkdir(exist_ok=True)
configs=[Config(),replace(Config(),stage='smoke',seeds=(3900,),n_train=64,n_val=64,n_test=128,total_steps=40,eval_interval=10),
         replace(Config(),stage='pilot',seeds=(3910,3911),total_steps=400,eval_interval=50)]
for cfg in configs:
    path=config/(cfg.stage+'.json')
    with path.open('x') as f: json.dump(asdict(cfg.validate()),f,indent=2);f.write('\n')
protected={}
for folder in ('conn_experiment_01','CoNN_Associative_Relationships','CoNN_Cursor_Experiment_01','AUDIT_20261020','ORGANIZED_EXPERIMENTS','docs'):
    for p in (ROOT/folder).rglob('*'):
        if p.is_file() and p.suffix.lower() in ('.py','.md','.mmd','.yaml','.yml','.toml','.json','.tla','.cfg'):
            if any(part.startswith('.') or part=='node_modules' for part in p.relative_to(ROOT).parts): continue
            if 'experiment_04' in p.parts: continue
            protected[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
with (base/'original_files_sha256.json').open('x') as f: json.dump(protected,f,indent=2)
print('Protected original source/document/config files:',len(protected))
for cfg in configs: print(cfg.stage,digest_json(asdict(cfg)))
