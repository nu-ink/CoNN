"""UCI HAR audit, label-independent layouts and train-subject-only normalization."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
import re
from zipfile import ZipFile

import numpy as np
import pandas as pd

from conn_exp.experiment1b import REPO, dump, seal, sha, tree_manifest, utc

DATASET_URL='https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip'
DATASET_PAGE='https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones'
CHANNELS=[f'{sensor}_{axis}' for sensor in ['body_acc','body_gyro','total_acc'] for axis in ['x','y','z']]
SPLIT_SEED=104729


def subject_partition(train_subjects,test_subjects):
    train_ids=np.unique(train_subjects)
    test_ids=np.unique(test_subjects)
    if set(train_ids)&set(test_ids):
        raise ValueError('Official subject sets overlap')
    val_ids=np.sort(np.random.default_rng(SPLIT_SEED).permutation(train_ids)[:5])
    fit_ids=np.setdiff1d(train_ids,val_ids)
    return fit_ids,val_ids,test_ids


def fit_scaler(train,representation):
    values=np.asarray(train,dtype=np.float64)
    if representation=='temporal':
        values=values.reshape(-1,128,9)
        mean=values.mean(axis=(0,1));scale=values.std(axis=(0,1))
    elif representation=='engineered':
        mean=values.mean(axis=0);scale=values.std(axis=0)
    else:
        raise ValueError('Unknown representation')
    constant=scale<1e-8
    scale=np.where(constant,1.,scale)
    return mean,scale,constant


def transform(values,mean,scale,representation):
    values=np.asarray(values,dtype=np.float64)
    if representation=='temporal':
        shape=values.shape
        values=((values.reshape(-1,128,9)-mean)/scale).reshape(shape)
    else:
        values=(values-mean)/scale
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite transformed input')
    return np.ascontiguousarray(values,dtype=np.float32)


def feature_layout(names,representation):
    if representation=='engineered':
        # Metadata index order is fixed without any feature-value or label input.
        return pd.DataFrame(dict(column=np.arange(len(names)),name=names,
            coordinate=np.linspace(0,1,len(names)),relation='official_metadata_order_semantic_not_physical'))
    if representation=='temporal':
        return pd.DataFrame([dict(column=t*9+c,name=f'{name}@sample_{t}',channel=name,
            sample=t,seconds=t/50,coordinate=t/127,relation='within_window_time_channels_colocated')
            for t in range(128) for c,name in enumerate(CHANNELS)])
    raise ValueError('Unknown representation')


def prepare(root,archive):
    root=Path(root).resolve();archive=Path(archive).resolve()
    root.mkdir(parents=True,exist_ok=False)
    audit=root/'audit';audit.mkdir()
    old=[p for p in sorted((REPO/'results/runs').iterdir()) if p.is_dir() and p!=root]
    dump(root/'prior_artifacts.before.json',{p.name:tree_manifest(p) for p in old})
    with ZipFile(archive) as outer:
        nested=outer.read('UCI HAR Dataset.zip')
    with ZipFile(BytesIO(nested)) as z:
        file_rows=[dict(path=i.filename,bytes=i.file_size,crc32=i.CRC) for i in z.infolist()]
        pd.DataFrame(file_rows).to_csv(audit/'archive_inventory.csv',index=False)
        prefix='UCI HAR Dataset/'
        def read(name):
            return z.read(prefix+name)
        def array(name,dtype=np.float32):
            return np.loadtxt(BytesIO(read(name)),dtype=dtype)
        for name in ['README.txt','features_info.txt','features.txt','activity_labels.txt']:
            (audit/name).write_bytes(read(name))
        labels={int(line.split()[0]):line.split()[1] for line in read('activity_labels.txt').decode().splitlines() if line.strip()}
        features=[line.split(maxsplit=1)[1] for line in read('features.txt').decode().splitlines() if line.strip()]
        feature_ids=[int(line.split(maxsplit=1)[0]) for line in read('features.txt').decode().splitlines() if line.strip()]
        assert feature_ids==list(range(1,562)) and len(features)==561
        raw={};signal_rows=[]
        for split in ['train','test']:
            x=array(f'{split}/X_{split}.txt')
            y=array(f'{split}/y_{split}.txt',np.int64)-1
            subjects=array(f'{split}/subject_{split}.txt',np.int64)
            assert x.shape==(len(y),561) and len(subjects)==len(y)
            assert np.isfinite(x).all() and set(y)==set(range(6))
            signals=[]
            for channel in CHANNELS:
                a=array(f'{split}/Inertial Signals/{channel}_{split}.txt')
                assert a.shape==(len(y),128) and np.isfinite(a).all()
                signals.append(a)
                signal_rows.append(dict(official_split=split,channel=channel,rows=a.shape[0],samples=a.shape[1],finite=True))
            # A time-major flattening gives all simultaneous channels one coordinate.
            temporal=np.stack(signals,axis=2).reshape(len(y),-1)
            raw[split]=dict(engineered=x,temporal=temporal,y=y,subject=subjects)
        pd.DataFrame(signal_rows).to_csv(audit/'signal_inventory.csv',index=False)
    train_ids,val_ids,test_ids=subject_partition(raw['train']['subject'],raw['test']['subject'])
    partition=dict(fit_subjects=train_ids.tolist(),validation_subjects=val_ids.tolist(),test_subjects=test_ids.tolist(),
        split_seed=SPLIT_SEED,rule='Keep official 9-subject test; choose 5 of 21 official train subjects with a fixed RNG permutation; remaining 16 fit. No labels or feature values enter assignment.')
    seal(root/'subject_split.frozen.json',partition)
    fit=np.isin(raw['train']['subject'],train_ids);validation=np.isin(raw['train']['subject'],val_ids)
    assert fit.sum()+validation.sum()==len(fit) and not (fit&validation).any()
    split_indices={'train':('train',np.flatnonzero(fit)),'val':('train',np.flatnonzero(validation)),
        'test':('test',np.arange(len(raw['test']['y'])))}
    split_rows=[]
    for name,(origin,index) in split_indices.items():
        source=raw[origin]
        for subject in np.unique(source['subject'][index]):
            subject_idx=index[source['subject'][index]==subject]
            for label in range(6):
                split_rows.append(dict(split=name,subject=int(subject),label=label+1,activity=labels[label+1],
                    windows=int((source['y'][subject_idx]==label).sum())))
    pd.DataFrame(split_rows).to_csv(audit/'subject_class_counts.csv',index=False)
    prepared=root/'prepared';prepared.mkdir()
    provenance={}
    for representation in ['engineered','temporal']:
        folder=prepared/representation;folder.mkdir()
        mean,scale,constant=fit_scaler(raw['train'][representation][fit],representation)
        np.savez(folder/'scaler.npz',mean=mean,scale=scale,constant=constant,fit_subjects=train_ids)
        layout=feature_layout(features,representation)
        layout.to_csv(folder/'feature_coordinates.csv',index=False)
        for name,(origin,index) in split_indices.items():
            source=raw[origin]
            x=transform(source[representation][index],mean,scale,representation)
            np.save(folder/f'x_{name}.npy',x)
            np.save(folder/f'y_{name}.npy',source['y'][index])
            np.save(folder/f'subject_{name}.npy',source['subject'][index])
        provenance[representation]=dict(input_features=len(layout),scaler_fitted_on_subjects=train_ids.tolist(),
            scaler_dimensions=len(mean),constant_dimensions=int(constant.sum()),
            scaler_rule='per-channel mean/std across training windows and time' if representation=='temporal' else 'per-feature mean/std across training windows',
            layout_rule=layout.relation.iloc[0])
    files={p.relative_to(prepared).as_posix():sha(p) for p in sorted(prepared.rglob('*')) if p.is_file()}
    dump(root/'prepared_manifest.json',files)
    split_sizes={name:len(index) for name,(_,index) in split_indices.items()}
    duplicate_names=int(pd.Series(features).duplicated().sum())
    audit_result=dict(completed_utc=utc(),dataset_url=DATASET_URL,dataset_page=DATASET_PAGE,
        dataset_doi='10.24432/C54S4K',archive_path=str(archive),archive_sha256=sha(archive),
        official_train_windows=len(raw['train']['y']),official_test_windows=len(raw['test']['y']),
        total_subjects=30,classes=labels,engineered_features=561,duplicate_feature_names=duplicate_names,
        duplicate_handling='Retain every column by original integer ID; never merge equal names.',
        signal_channels=CHANNELS,window_samples=128,sampling_hz=50,window_overlap_fraction=.5,
        split_sizes=split_sizes,partition=partition,representations=provenance,
        test_access='Schema, finite values and subject/class counts audited. No model predictions or model selection on test.',
        vendor_preprocessing_limitation='561 features are supplied normalized to [-1,1]; README does not identify which split fitted original normalization. Our normalization is train-subject-only and cannot undo unknown upstream choices. Temporal signals are also vendor-filtered windows, not untouched continuous recordings.',
        timing_limitation='No complete session/timestamp metadata: within-window temporal order is available, continuous ordering across rows is not assumed.',
        source_documentation='Official UCI page and archive README/features_info; archive documentation retained verbatim for provenance.',
        license_note='Current UCI page lists CC BY 4.0; bundled 2013 README contains older noncommercial wording. Both source records are retained; this run performs research analysis without redistributing a modified dataset.',
        missing_values=False,models_trained=0)
    dump(audit/'dataset_audit.json',audit_result)
    lines=['# Experiment 2: UCI HAR dataset audit','',
        f'Source: [UCI HAR]({DATASET_PAGE}), DOI 10.24432/C54S4K. Archive SHA-256 `{sha(archive)}`. Metadata and archive inventory are retained alongside this report.','',
        f"Measured structure: {audit_result['official_train_windows']} official training windows, {audit_result['official_test_windows']} official test windows, 30 subjects, six activity labels, 561 engineered columns and nine 128-sample sensor channels. Duplicate engineered names: {duplicate_names}; all columns are retained by integer ID.",
        'Activities are walking, walking upstairs, walking downstairs, sitting, standing and laying. Subject IDs are metadata, never input features. Train/test official subject sets are disjoint. Features describe acceleration, gravity/body motion, gyroscope, jerk, magnitude, frequency statistics and angles; original feature names and all sensor channel names are saved.','',
        '## Predetermined subject-disjoint partitions','',f'Fit subjects: {train_ids.tolist()}.',f'Validation subjects: {val_ids.tolist()}.',f'Test subjects: {test_ids.tolist()}.',f'Window counts: {split_sizes}.',
        'The fixed split seed selects validation subjects from official training IDs only. No class balance search, feature inspection or test outcomes select the split. Per-subject/class counts are structural audit measurements; test performance remains unmeasured.','',
        '## Coordinates and leakage controls','',
        'Primary representation: the released 561 engineered columns, in their original indexed order, with coordinate j/560. This is a semantic file-order prior, not a measured physical distance or a temporal sample axis. No covariance, labels or model outcomes define the layout.',
        'Secondary representation: nine inertial channels flattened time-major, coordinate t/127 for all nine channels at time sample t. Channels at the same instant are colocated; the X/Y/Z axes are not assigned invented physical distances. Hidden-unit coordinates are fixed uniform unit-interval points. This encodes within-window temporal proximity only.',
        'All added scaling is fitted once on the 16 training subjects: per engineered column, or per raw sensor channel across training windows/time. Validation/test receive those frozen training statistics. No feature removal, missing-value imputation, label-informed coordinate learning or split-specific renormalization is used. Overlapping windows cannot cross subject partitions.',
        audit_result['vendor_preprocessing_limitation'],audit_result['timing_limitation'],'',
        'The primary and secondary layouts are specified before training and are never selected using test accuracy. Native feature values are not permuted in the coordinate control; only the assignment of input coordinates changes.','',
        '## Scope and source limitations','',audit_result['license_note'],
        'This audit is descriptive, including test schema and counts. It performs no model evaluation. Independent seed repetitions on this one fixed subject partition will quantify initialization/order variation, not variability across new subject partitions.','']
    (audit/'dataset_report.md').write_text('\n'.join(lines),encoding='utf-8')
    seal(audit/'completed.json',dict(completed_utc=utc(),models_trained=0,test_model_evaluations=0,
        report_sha256=sha(audit/'dataset_report.md'),prepared_manifest_sha256=sha(root/'prepared_manifest.json')))
    print(f'DATA AUDIT COMPLETE: {root}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--archive',type=Path,default=REPO/'data/raw/uci_har_original.zip')
    parser.add_argument('--run-dir',type=Path)
    args=parser.parse_args()
    root=args.run_dir or REPO/'results/runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_experiment2_har_v1')
    prepare(root,args.archive)
