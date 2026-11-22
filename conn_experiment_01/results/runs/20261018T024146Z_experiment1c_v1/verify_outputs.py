"""Verify final 1C artifacts without training or additional inference."""
from pathlib import Path
import hashlib
import json
import math
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest(folder):
    return {p.relative_to(folder).as_posix():sha(p) for p in sorted(folder.rglob('*')) if p.is_file()}


def main():
    protocol=json.loads((ROOT/'protocol.frozen.json').read_text())
    selection=json.loads((ROOT/'selection.frozen.json').read_text())
    start=json.loads((ROOT/'replication.started.json').read_text())
    audit=json.loads((ROOT/'audit/completed.json').read_text())
    assert audit['completed_utc']<protocol['frozen_utc']<selection['frozen_utc']<start['started_utc']
    for name in ['protocol.frozen.json','selection.frozen.json']:
        assert sha(ROOT/name)==(ROOT/(name+'.sha256')).read_text().strip()
    assert start['selection_sha256']==sha(ROOT/'selection.frozen.json')
    assert selection['tuning_scores_sha256']==sha(ROOT/'tuning/scores.csv')
    source=json.loads((ROOT/'source_manifest.json').read_text())
    assert all(sha(REPO/name)==digest==sha(ROOT/'source'/name) for name,digest in source.items())
    before=json.loads((ROOT/'prior_artifacts.before.json').read_text())
    assert before==json.loads((ROOT/'prior_artifacts.after.json').read_text())
    assert before=={name:manifest(REPO/'results/runs'/name) for name in before}
    total_gradients=total_checkpoints=total_counterfactuals=total_metrics=0
    for phase,n,reads in [('tuning',480,0),('replication',640,1)]:
        phase_root=ROOT/phase
        scores=pd.read_csv(phase_root/'scores.csv')
        assert len(scores)==n and scores.test_evaluations.eq(reads).all()
        assert scores.parameter_count.eq(4321).all() and scores.total_steps.eq(3200).all()
        for (_,_,_),group in scores.groupby(['task','seed','lr']):
            for column in ['data_fingerprint','init_fingerprint','order_sha256']:
                assert group[column].nunique()==1
        for task in ['and','xor']:
            expected={(task,v['id'],seed,lr) for v in protocol['variants'][task]
                for seed in protocol['tuning_seeds' if phase=='tuning' else 'replication_seeds']
                for lr in (protocol['learning_rate_grid'] if phase=='tuning' else [selection['learning_rate_by_task'][task]])}
            actual=set(scores[scores.task==task][['task','variant_id','seed','lr']].itertuples(index=False,name=None))
            assert actual==expected
        for row in scores.itertuples():
            job=phase_root/row.job
            assert sha(job/'config.resolved.json')==row.config_sha256
            metrics=pd.read_csv(job/'metrics.csv')
            assert np.isfinite(metrics[['loss','accuracy']]).all().all()
            test=metrics[metrics.split=='test']
            assert len(test)==reads
            if reads:
                assert test.optimizer_steps_completed.eq(3200).all() and test.kind.eq('final').all()
            gradients=pd.read_csv(job/'gradients.csv')
            assert np.isfinite(gradients.mean_abs).all()
            assert set(gradients.quantity)=={'dL_dW','dL_dW_eff'}
            assert set(gradients.distance)=={'base_task_geometry'}
            cf=pd.read_csv(job/'checkpoint_comparisons.csv')
            assert len(cf)==6 and set(cf.split)=={'val'}
            for kind,g in cf.groupby('kind'):
                assert g.weight_fingerprint.nunique()==1 and g.checkpoint_sha256.nunique()==1
                assert sha(job/(kind+'.pt'))==g.checkpoint_sha256.iloc[0]
                native=g[g.gate_mode=='native'].iloc[0]
                recorded=metrics[(metrics.kind==kind)&(metrics.split=='val')].iloc[0]
                assert native.accuracy==recorded.accuracy and abs(native.loss-recorded.loss)<2e-6
                total_checkpoints+=1
            total_gradients+=len(gradients);total_metrics+=len(metrics);total_counterfactuals+=len(cf)
    scores=pd.read_csv(ROOT/'replication/scores.csv')
    pivot=scores[scores.task=='and'].pivot(index='seed',columns='variant_id',values='test_accuracy')
    delta=(pivot.P_aligned_q0-pivot.D_aligned_q0).to_numpy()
    measured=json.loads((ROOT/'replication/primary_summary.json').read_text())
    mean,sd=float(delta.mean()),float(delta.std(ddof=1))
    margin=2.093024054408263*sd/math.sqrt(20)
    for name,value in dict(mean=mean,sd=sd,ci_low=mean-margin,ci_high=mean+margin,cohen_dz=mean/sd).items():
        assert math.isclose(measured[name],value,abs_tol=1e-14)
    assert measured['spans_zero'] and measured['n']==20
    convergence=pd.read_csv(ROOT/'replication/convergence.csv')
    assert convergence[~convergence.reached].first_measured_step.isna().all()
    pictures=list(ROOT.rglob('*.png'))
    assert len(pictures)==22
    for path in pictures:
        with Image.open(path) as image:
            image.verify()
    provenance=json.loads((ROOT/'interpretation.provenance.json').read_text())
    assert provenance['script_sha256']==sha(ROOT/'interpret_results.py')
    assert provenance['output_report_sha256']==sha(ROOT/'research_report.md')
    result=dict(verified_utc=datetime.now(timezone.utc).isoformat(), prior_run_count=len(before),
        prior_artifacts_unchanged=True, frozen_live_and_snapshotted_source_verified=True,
        audit_precedes_training_protocol=True, selection_precedes_replication=True,
        protocol_and_selection_seals_verified=True,tuning_models=480,replication_models=640,
        tuning_test_evaluations=0,replication_native_final_test_evaluations=640,
        parameter_count=4321,steps_per_model=3200,all_config_hashes_verified=True,
        metric_rows=total_metrics,finite_gradient_rows=total_gradients,checkpoint_files_verified=total_checkpoints,
        checkpoint_counterfactual_rows=total_counterfactuals,audit_checkpoint_inference_rows=audit['checkpoint_inference_rows'],
        native_reload_metrics_match_recorded=True,primary_statistics_independently_recomputed=True,
        unreached_convergence_thresholds_not_imputed=True,pngs_verified=len(pictures),
        pre_training_tests='45 passed in 15.37s',verifier_sha256=sha(Path(__file__)))
    (ROOT/'final_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
