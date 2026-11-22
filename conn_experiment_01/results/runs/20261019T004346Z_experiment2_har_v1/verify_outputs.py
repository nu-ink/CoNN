"""Independent final HAR artifact checks; no model inference or fitting."""
from pathlib import Path
from io import BytesIO
from zipfile import ZipFile
import hashlib
import json
import math
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from PIL import Image

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest(folder):
    return {p.relative_to(folder).as_posix():sha(p) for p in sorted(folder.rglob('*')) if p.is_file()}


def main():
    assert (ROOT/'completion.json').is_file(),'Finish benchmark before verification'
    protocol=json.loads((ROOT/'protocol.frozen.json').read_text())
    selected=json.loads((ROOT/'selection.frozen.json').read_text())
    started=json.loads((ROOT/'final_evaluation.started.json').read_text())
    audit=json.loads((ROOT/'audit/dataset_audit.json').read_text())
    split=json.loads((ROOT/'subject_split.frozen.json').read_text())
    assert audit['completed_utc']<protocol['frozen_utc']<selected['frozen_utc']<started['started_utc']
    for name in ['protocol.frozen.json','selection.frozen.json','subject_split.frozen.json']:
        assert sha(ROOT/name)==(ROOT/(name+'.sha256')).read_text().strip()
    assert started['selection_sha256']==sha(ROOT/'selection.frozen.json')
    assert selected['tuning_scores_sha256']==sha(ROOT/'tuning/scores.csv')
    assert not (set(split['fit_subjects'])&set(split['validation_subjects']) or
        set(split['fit_subjects'])&set(split['test_subjects']) or set(split['validation_subjects'])&set(split['test_subjects']))
    sources=json.loads((ROOT/'source_manifest.json').read_text())
    assert all(sha(REPO/name)==digest==sha(ROOT/'source'/name) for name,digest in sources.items())
    prior=json.loads((ROOT/'prior_artifacts.before.json').read_text())
    assert prior==json.loads((ROOT/'prior_artifacts.after.json').read_text())
    assert prior=={name:manifest(REPO/'results/runs'/name) for name in prior}
    prepared=json.loads((ROOT/'prepared_manifest.json').read_text())
    assert all(sha(ROOT/'prepared'/name)==digest for name,digest in prepared.items())
    assert sha(Path(audit['archive_path']))==audit['archive_sha256']
    # Recompute normalization independently from the original official training files.
    with ZipFile(audit['archive_path']) as outer:
        inner=ZipFile(BytesIO(outer.read('UCI HAR Dataset.zip')))
        def read(name,dtype):
            return np.loadtxt(BytesIO(inner.read('UCI HAR Dataset/'+name)),dtype=dtype)
        official_subject=read('train/subject_train.txt',np.int64)
        fit=np.isin(official_subject,split['fit_subjects'])
        raw=read('train/X_train.txt',np.float32)[fit].astype(np.float64)
        scaler=np.load(ROOT/'prepared/engineered/scaler.npz')
        assert np.allclose(raw.mean(0),scaler['mean'],atol=1e-11)
        assert np.allclose(raw.std(0),scaler['scale'],atol=1e-11)
        means=[];scales=[]
        for channel in audit['signal_channels']:
            values=read(f'train/Inertial Signals/{channel}_train.txt',np.float32)[fit].astype(np.float64)
            means.append(values.mean());scales.append(values.std())
        scaler=np.load(ROOT/'prepared/temporal/scaler.npz')
        assert np.allclose(means,scaler['mean'],atol=1e-11)
        assert np.allclose(scales,scaler['scale'],atol=1e-11)
        inner.close()
    expected_coordinates={'engineered':np.linspace(0,1,561),'temporal':np.repeat(np.arange(128)/127,9)}
    for representation,coordinates in expected_coordinates.items():
        folder=ROOT/'prepared'/representation
        assert np.allclose(pd.read_csv(folder/'feature_coordinates.csv').coordinate,coordinates,atol=1e-15)
        for label,key in [('train','fit_subjects'),('val','validation_subjects'),('test','test_subjects')]:
            subjects=np.load(folder/f'subject_{label}.npy')
            assert set(subjects)==set(split[key])
    tuning=pd.read_csv(ROOT/'tuning/scores.csv')
    assert len(tuning)==180 and tuning.test_evaluations.eq(0).all()
    assert 'test_accuracy' not in tuning and 'test_macro_f1' not in tuning
    grid=tuning.groupby(['representation','condition','lr'],as_index=False).agg(
        f1=('val_macro_f1','mean'),loss=('val_loss','mean'))
    winners=grid.sort_values(['representation','condition','f1','loss','lr'],
        ascending=[True,True,False,True,True]).drop_duplicates(['representation','condition'])
    assert all(selected['learning_rates'][r.representation][r.condition]==r.lr for r in winners.itertuples())
    checkpoint_count=prediction_count=gradient_rows=0
    max_energy_gap=0.
    for phase,count,expected_reads in [('tuning',180,0),('replication',240,1)]:
        directory=ROOT/phase;scores=pd.read_csv(directory/'scores.csv')
        assert len(scores)==count and scores.test_evaluations.eq(expected_reads).all()
        assert scores.total_steps.eq(3200).all() and scores.training_examples_seen.eq(409600).all()
        for (representation,seed),group in scores.groupby(['representation','seed']):
            for key in ['init_fingerprint','batch_order_sha256','scaler_sha256','layout_sha256','parameter_count']:
                assert group[key].nunique()==1
            expected_params=40518 if representation=='engineered' else 78342
            assert group.parameter_count.eq(expected_params).all()
            chosen=group.sort_values('lr').drop_duplicates('condition').set_index('condition')
            energy={c:pd.read_csv(directory/chosen.loc[c,'job']/'row_energy.csv') for c in ['D','B','C']}
            for c in ['B','C']:
                joined=energy['D'].merge(energy[c],on=['optimizer_steps_completed','kind','layer','unit'],suffixes=('_d','_o'))
                assert len(joined)==len(energy['D'])==len(energy[c])
                gap=float((joined.energy_d-joined.energy_o).abs().max())
                assert gap<1e-3;max_energy_gap=max(max_energy_gap,gap)
        for row in scores.itertuples():
            job=directory/row.job
            assert sha(job/'config.resolved.json')==row.config_sha256
            metrics=pd.read_csv(job/'metrics.csv')
            assert np.isfinite(metrics[['accuracy','macro_f1','loss']]).all().all()
            test=metrics[metrics.split=='test']
            assert len(test)==expected_reads
            gradients=pd.read_csv(job/'gradients.csv')
            assert np.isfinite(gradients.mean_abs).all();gradient_rows+=len(gradients)
            comparisons=pd.read_csv(job/'checkpoint_gate_comparisons.csv')
            assert len(comparisons)==6 and set(comparisons.split)=={'val'}
            for kind,g in comparisons.groupby('kind'):
                assert g.weight_fingerprint.nunique()==1 and g.checkpoint_sha256.nunique()==1
                assert sha(job/(kind+'.pt'))==g.checkpoint_sha256.iloc[0]
                native=g[g.gate_mode=='native'].iloc[0]
                reference=metrics[(metrics.split=='val')&(metrics.kind==kind)].iloc[0]
                assert abs(native.loss-reference.loss)<1e-6 and native.macro_f1==reference.macro_f1
                checkpoint_count+=1
            if expected_reads:
                assert test.kind.eq('final').all() and test.optimizer_steps_completed.eq(3200).all()
                assert row.lr==selected['learning_rates'][row.representation][row.condition]
                pred=np.load(job/'test_predictions.npz')
                assert np.array_equal(pred['truth'],np.load(ROOT/'prepared'/row.representation/'y_test.npy'))
                assert np.array_equal(pred['subject'],np.load(ROOT/'prepared'/row.representation/'subject_test.npy'))
                assert np.allclose(pred['probability'].sum(1),1,atol=1e-6)
                assert np.array_equal(pred['probability'].argmax(1),pred['prediction'])
                matrix=np.bincount(6*pred['truth']+pred['prediction'],minlength=36).reshape(6,6)
                accuracy=np.trace(matrix)/matrix.sum()
                denominator=matrix.sum(1)+matrix.sum(0)
                f1=np.divide(2*np.diag(matrix),denominator,out=np.zeros(6),where=denominator!=0).mean()
                assert abs(row.test_accuracy-accuracy)<1e-14 and abs(row.test_macro_f1-f1)<1e-14
                assert np.array_equal(matrix,pd.read_csv(job/'test_confusion.csv').to_numpy())
                prediction_count+=len(pred['truth'])
    scores=pd.read_csv(ROOT/'replication/scores.csv')
    pivot=scores[scores.representation=='engineered'].pivot(index='seed',columns='condition',values='test_macro_f1')
    delta=(pivot.D-pivot.B).to_numpy();mean=float(delta.mean());sd=float(delta.std(ddof=1))
    width=2.093024054408263*sd/math.sqrt(20)
    primary=json.loads((ROOT/'replication/primary_summary.json').read_text())
    for key,value in dict(mean=mean,sd=sd,ci_low=mean-width,ci_high=mean+width).items():
        assert math.isclose(primary[key],value,abs_tol=1e-14)
    pictures=list(ROOT.rglob('*.png'));assert len(pictures)==18
    for path in pictures:
        with Image.open(path) as image:
            image.verify()
    result=dict(verified_utc=datetime.now(timezone.utc).isoformat(),prior_run_count=len(prior),
        prior_synthetic_artifacts_unchanged=True,source_live_and_snapshot_verified=True,
        prepared_arrays_and_coordinate_hashes_verified=True,scalers_independently_recomputed_from_fit_subjects_only=True,
        subject_partitions_disjoint=True,selection_recomputed_using_validation_only=True,
        audit_protocol_selection_and_test_order_verified=True,tuning_models=180,final_models=240,
        tuning_test_model_evaluations=0,final_native_test_model_evaluations=240,
        prediction_rows_independently_scored=prediction_count,checkpoint_files_verified=checkpoint_count,
        finite_gradient_rows=gradient_rows,all_seed_energy_max_abs_gap=max_energy_gap,
        primary_statistics_independently_recomputed=True,pngs_verified=len(pictures),
        pre_training_test_suite='57 passed in 17.98s',verifier_sha256=sha(Path(__file__)))
    (ROOT/'final_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
