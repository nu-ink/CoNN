"""Independent file/prediction/budget checks after the complete development grid."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path

import numpy as np
import torch

from conn_assoc.paths import EXPERIMENT_ROOT
from conn_assoc.phase23_core import configure,fingerprint,json_write,metrics,sha


def verify(run):
    configure();config=json.loads((run/'config.frozen.json').read_text())
    report={'verified_utc':datetime.now(timezone.utc).isoformat(),'fits':0,'optimizer_updates':0,
            'checkpoint_files':0,'prediction_rows_rescored':0,'finite_gradient_records':0,
            'pre_post_release_weight_pairs_identical':0,'official_test_model_evaluations':0}
    seen=set();initial={};row_sets={};validation_subjects=set()
    for condition in config['conditions']:
        for seed in config['seeds']:
            for fold in range(7):
                for lr in config['learning_rates']:
                    job=run/'jobs'/f'{condition}_seed{seed}_fold{fold}_lr{lr:g}'
                    assert (job/'completion.json').exists(),job
                    complete=json.loads((job/'completion.json').read_text())
                    assert complete['completed_steps']==3200 and complete['official_test_evaluations']==0
                    assert complete['checkpoint_reload_exact']
                    report['fits']+=1;report['optimizer_updates']+=complete['completed_steps']
                    for name,digest in complete['files'].items():
                        assert sha(job/name)==digest,(job,name)
                        report['checkpoint_files']+=name.endswith('.pt')
                    spec=json.loads((job/'config.json').read_text())
                    key=(condition,seed,fold,lr);assert key not in seen;seen.add(key)
                    record=json.loads((job/'final_metrics.json').read_text())
                    with np.load(job/'predictions.npz') as z:
                        assert set(z['val_subject'])==set(config['folds'][fold])
                        assert set(z['train_subject']).isdisjoint(set(z['val_subject']))
                        assert set(z['val_subject']).isdisjoint({2,4,9,10,12,13,18,20,24})
                        assert set(z['train_subject']).isdisjoint({2,4,9,10,12,13,18,20,24})
                        validation_subjects.update(int(s) for s in z['val_subject'])
                        fit_rows=set(z['train_row']);val_rows=set(z['val_row'])
                        assert fit_rows.isdisjoint(val_rows) and fit_rows|val_rows==set(range(7352))
                        if fold in row_sets:assert row_sets[fold]==val_rows
                        row_sets[fold]=val_rows
                        for split in ('train','val'):
                            for row in record['scores'][split]:
                                result=metrics(z[split+'_truth'],z[split+'_context'+str(row['context'])])
                                report['prediction_rows_rescored']+=len(z[split+'_truth'])
                                assert result['confusion']==row['confusion']
                                for name in ('accuracy','macro_f1'):assert abs(result[name]-row[name])<1e-12
                                assert np.allclose(result['recall'],row['recall'],atol=1e-12,rtol=0)
                    for g in json.loads((job/'gradients.json').read_text()):
                        assert np.isfinite(list(g['norms'].values())).all()
                        assert np.isfinite(g['loss']);report['finite_gradient_records']+=1
                    pre=torch.load(job/'pre_release.pt',map_location='cpu',weights_only=False)
                    post=torch.load(job/'post_release.pt',map_location='cpu',weights_only=False)
                    assert pre['next_step']==post['next_step']==2240
                    assert fingerprint(pre['model'])==fingerprint(post['model'])
                    assert np.array_equal(pre['batches'],post['batches'])
                    report['pre_post_release_weight_pairs_identical']+=1
                    start=torch.load(job/'initial.pt',map_location='cpu',weights_only=False)
                    base={k:v for k,v in start['model'].items() if not k.startswith('context_')}
                    signature=fingerprint(base)
                    if seed in initial:assert initial[seed]==signature
                    initial[seed]=signature
                    assert complete['initial_fingerprint']!=complete['final_fingerprint']
    assert len(seen)==1050 and len(validation_subjects)==21
    assert set.union(*row_sets.values())==set(range(7352))
    assert sum(map(len,row_sets.values()))==7352
    prior=json.loads((run/'experiment2.before.json').read_text())
    changed=[p for p,r in prior.items() if sha(p)!=r['sha256']]
    assert not changed,changed
    protected=json.loads((run/'prior_associative_manifest.json').read_text())
    changed_assoc=[p for p,h in protected.items() if sha(EXPERIMENT_ROOT/p)!=h]
    assert not changed_assoc,changed_assoc
    cpu_before=(run/'cpu_pip_freeze.log').read_text().splitlines()
    cpu_after=(run/'cpu_requirements.after_install.txt').read_text().splitlines()
    assert cpu_before==cpu_after
    report.update(original_experiment2_files_unchanged=len(prior),prior_associative_files_unchanged=len(protected),
        cpu_environment_requirements_unchanged=True,paired_base_initialization_verified=True,
        validation_fold_row_and_subject_isolation_verified=True,amendment_sha256=sha(EXPERIMENT_ROOT/'PROTOCOL_PHASE23_v1.md'),
        all_grid_hashes_verified=True)
    json_write(run/'final_verification.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();verify(a.run)
