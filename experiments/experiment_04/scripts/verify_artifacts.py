"""Verify original preservation, pairing, checkpoints, saved predictions and result hashes."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from conn_exp04 import ROOT
from conn_exp04.core import Config, CONDITIONS, TASKS, arrays_hash, digest_json, metrics


def main(run, label=''):
    base=ROOT/'experiments'/'experiment_04'
    original=json.loads((base/'original_files_sha256.json').read_text())
    changed=[p for p,h in original.items() if not (ROOT/p).is_file() or hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
    if changed: raise AssertionError(f'Original files changed: {changed}')
    cfg=Config.load(run/'config.json')
    env=json.loads((run/'environment.json').read_text())
    for p,h in env['source_hashes'].items():
        assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h, p
    rows=json.loads((run/'evaluation'/'metrics.json').read_text())
    checks=0
    checkpoints=0
    for s in cfg.seeds:
        for task in TASKS:
            entries=[]
            data=np.load(run/'evaluation'/f'{s}_{task}_test.npz')
            data_hash=arrays_hash(data['x'],data['y'])
            for c in CONDITIONS:
                d=run/f'seed_{s}'/task/c
                status=json.loads((d/'status.json').read_text())
                assert status['state']=='trained' and status['completed_steps']==cfg.total_steps
                provenance=json.loads((d/'provenance.json').read_text())
                entries.append(provenance)
                prediction=np.load(run/'evaluation'/f'{s}_{task}_{c}_predictions.npz')
                assert np.array_equal(prediction['y'],data['y'])
                for endpoint in ('pre_release','immediate_release','final'):
                    row=next(r for r in rows if (r['seed'],r['task'],r['condition'],r['endpoint'])==(s,task,c,endpoint))
                    recalculated=metrics(torch.from_numpy(prediction[endpoint]),torch.from_numpy(data['y']))
                    assert row['test_sha256']==data_hash
                    for key in ('accuracy','f1','loss'): assert abs(row[key]-recalculated[key])<1e-7
                    checks+=1
                for name in ('pre_release.pt','final.pt'):
                    ck=torch.load(d/name,map_location='cpu',weights_only=True)
                    assert ck['config_sha256']==digest_json(asdict(cfg))
                    assert ck['initialization_sha256']==provenance['initialization_sha256']
                    assert all(torch.isfinite(t).all() for t in ck['model'].values())
                    assert sum(t.numel() for t in ck['model'].values())==4321
                    if name=='final.pt': assert all(torch.equal(g,torch.ones_like(g)) for g in ck['gates'])
                    checkpoints+=1
            for key in ('parameters','initialization_sha256','batch_sha256','optimizer','data_hashes'):
                assert all(e[key]==entries[0][key] for e in entries), key
    # Catalog covers all run files at this verification time, including the source snapshot.
    suffix=('_'+label) if label else ''
    manifest_path=run/('artifact_sha256'+suffix+'.json')
    integrity_path=run/('integrity'+suffix+'.json')
    if manifest_path.exists() or integrity_path.exists():
        raise FileExistsError('Use a new verification label; prior verification artifacts are preserved')
    hashes={str(p.relative_to(run)):hashlib.sha256(p.read_bytes()).hexdigest() for p in run.rglob('*') if p.is_file()}
    with manifest_path.open('x') as f: json.dump(hashes,f,indent=2);f.write('\n')
    report={'original_files_checked':len(original),'original_files_changed':changed,
            'source_hashes_match':True,'saved_endpoint_metrics_recomputed':checks,
            'checkpoints_checked':checkpoints,'fit_pairing_verified':True,
            'artifact_files_hashed':len(hashes),'formal_model_checking':False}
    with integrity_path.open('x') as f: json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main(Path(sys.argv[1]).resolve(),sys.argv[2] if len(sys.argv)>2 else '')
