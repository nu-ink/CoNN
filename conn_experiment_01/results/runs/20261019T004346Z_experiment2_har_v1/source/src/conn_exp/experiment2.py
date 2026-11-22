"""UCI HAR: equal tuning, freeze, then subject-disjoint final evaluation."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
import yaml

from conn_exp.experiment1b import REPO,dump,seal,sha,tree_manifest,utc
from conn_exp.gates import draw_masks
from conn_exp.har_core import CONDITION_NAMES,HARGates,geometry_from_coordinates,initial_state,paired_batches,train
from conn_exp.runtime import configure_runtime


def load_config(path):
    config=yaml.safe_load(Path(path).read_text())
    if config['optimizer']!='sgd' or config['momentum']!=0 or config['weight_decay']!=0 or config['device']!='cpu':
        raise ValueError('This frozen benchmark uses CPU plain SGD without momentum/decay')
    if config['conditions']!=['A','S08','S25','D','B','C'] or config['representations']!=['engineered','temporal']:
        raise ValueError('Benchmark conditions/representations must be retained')
    if set(config['tuning_seeds'])&set(config['replication_seeds']):
        raise ValueError('Tuning/replication seed overlap')
    return config


def select_validation(scores,config):
    keys=['representation','condition','lr','seed']
    frame=scores[keys+['val_macro_f1','val_loss']].copy()
    if frame.duplicated(keys).any() or not np.isfinite(frame[['val_macro_f1','val_loss']]).all().all():
        raise ValueError('Duplicate or nonfinite tuning result')
    expected={(r,c,lr,seed) for r in config['representations'] for c in config['conditions']
        for lr in config['learning_rates'] for seed in config['tuning_seeds']}
    if set(frame[keys].itertuples(index=False,name=None))!=expected:
        raise ValueError('Incomplete or unequal tuning budget')
    grid=frame.groupby(keys[:3],as_index=False).agg(mean_val_macro_f1=('val_macro_f1','mean'),
        mean_val_loss=('val_loss','mean'),n=('seed','size'))
    best=grid.sort_values(['representation','condition','mean_val_macro_f1','mean_val_loss','lr'],
        ascending=[True,True,False,True,True]).drop_duplicates(['representation','condition'])
    selection={r:{row.condition:float(row.lr) for row in best[best.representation==r].itertuples()}
        for r in config['representations']}
    return grid,selection


def worker(root_string,phase,representation,seed,config,rates):
    root=Path(root_string);phase_root=root/phase
    configure_runtime()
    folder=root/'prepared'/representation
    arrays={name:np.load(folder/(name+'.npy')) for name in ['x_train','y_train','x_val','y_val']}
    if phase=='replication':
        frozen=json.loads((root/'selection.frozen.json').read_text())
        assert sha(root/'selection.frozen.json')==(root/'selection.frozen.json.sha256').read_text().strip()
        assert rates=={c:[lr] for c,lr in frozen['learning_rates'][representation].items()}
        arrays.update({name:np.load(folder/(name+'.npy')) for name in ['x_test','y_test','subject_test']})
    layout=pd.read_csv(folder/'feature_coordinates.csv')
    geometry=geometry_from_coordinates(layout.coordinate.to_numpy(),config['hidden_sizes'])
    init=initial_state(arrays['x_train'].shape[1],config['hidden_sizes'],seed)
    masks=draw_masks(arrays['x_train'].shape[1],*config['hidden_sizes'],seed)
    batches=paired_batches(len(arrays['y_train']),config['total_steps'],config['batch_size'],seed)
    paired=phase_root/'paired_inputs'/representation;paired.mkdir(parents=True,exist_ok=True)
    np.save(paired/f'seed_{seed}_batches.npy',batches)
    torch.save(dict(initial_state=init,masks=vars(masks)),paired/f'seed_{seed}.pt')
    # Execution order is independent from model initialization and minibatch streams.
    order=np.random.default_rng(seed+4_000_003).permutation(config['conditions']).tolist()
    rows=[]
    journal=phase_root/'journals'/f'{representation}_seed_{seed}.jsonl'
    for condition in order:
        for lr in rates[condition]:
            job=phase_root/'jobs'/representation/f'seed_{seed}'/condition/f'lr_{lr:g}'
            started=time.perf_counter()
            score=train(config,representation,condition,seed,lr,arrays,init,masks,geometry,batches,job,
                final_test=phase=='replication')
            score['job_elapsed_wall_seconds']=time.perf_counter()-started
            score.update(job=job.relative_to(phase_root).as_posix(),execution_order=order.index(condition),
                scaler_sha256=sha(folder/'scaler.npz'),layout_sha256=sha(folder/'feature_coordinates.csv'),
                prepared_manifest_sha256=sha(root/'prepared_manifest.json'))
            with journal.open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(score,allow_nan=False)+'\n')
            rows.append(score)
            print(f'{phase} {representation} seed={seed} {condition} lr={lr:g} complete',flush=True)
    return rows


def audit_phase(root,scores):
    for (_,seed),group in scores.groupby(['representation','seed']):
        for field in ['init_fingerprint','batch_order_sha256','scaler_sha256','layout_sha256','parameter_count','total_steps']:
            assert group[field].nunique()==1,field
    # The gate schedule does not depend on LR or weights, so one complete seed/cell
    # per representation verifies all schedule points; all initialization/order hashes
    # above are checked for every fit.
    max_gap=0.
    for representation,group in scores.groupby('representation'):
        selected=group[group.seed==group.seed.min()].sort_values('lr').drop_duplicates('condition')
        energy={r.condition:pd.read_csv(root/r.job/'row_energy.csv') for r in selected.itertuples()}
        for condition in ['B','C']:
            joined=energy['D'].merge(energy[condition],on=['optimizer_steps_completed','kind','layer','unit'],suffixes=('_d','_o'))
            assert len(joined)==len(energy['D'])==len(energy[condition])
            gap=float((joined.energy_d-joined.energy_o).abs().max())
            max_gap=max(max_gap,gap);assert gap<1e-3
    dump(root/'invariant_audit.json',dict(all_fit_pairings_checked=True,paired_initialization_batches_and_scaler=True,
        constant_parameter_count_within_representation=True,energy_schedule_checked_representations=2,
        gate_energy_max_abs_gap=max_gap,test_model_evaluations=int(scores.test_evaluations.sum())))


def run_phase(root,phase,config,rates,workers):
    directory=root/phase;directory.mkdir(exist_ok=False);(directory/'journals').mkdir()
    seeds=config['tuning_seeds' if phase=='tuning' else 'replication_seeds']
    rows=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(worker,str(root),phase,representation,seed,config,rates[representation])
            for seed in seeds for representation in config['representations']]
        for future in as_completed(futures):
            rows.extend(future.result())
            pd.DataFrame(rows).to_csv(directory/'scores.csv',index=False)
            print(f'BATCH COMPLETE {phase}: {len(rows)} fitted models aggregated',flush=True)
    frame=pd.DataFrame(rows).sort_values(['representation','seed','condition','lr'])
    frame.to_csv(directory/'scores.csv',index=False)
    audit_phase(directory,frame)
    return frame


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=REPO/'configs/har.yaml')
    parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args(argv)
    root=args.run_dir.resolve();config=load_config(args.config)
    if not (root/'audit/completed.json').is_file() or (root/'protocol.frozen.json').exists():
        raise RuntimeError('Use a fresh, completed dataset-audit directory')
    if not 1<=args.workers<=(os.cpu_count() or 1):
        raise ValueError('Invalid worker count')
    flags=configure_runtime()
    prepared=json.loads((root/'prepared_manifest.json').read_text())
    assert all(sha(root/'prepared'/name)==digest for name,digest in prepared.items())
    split=json.loads((root/'subject_split.frozen.json').read_text())
    for representation in config['representations']:
        for name in ['train','val','test']:
            assert set(np.unique(np.load(root/'prepared'/representation/f'y_{name}.npy')))==set(range(6))
    files=[p for folder in ['src','tests','configs'] for p in (REPO/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    files += [REPO/'README.md',REPO/'DESIGN_DECISIONS.md',REPO/'pyproject.toml']
    source={}
    for path in sorted(files):
        relative=path.relative_to(REPO);target=root/'source'/relative
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        source[relative.as_posix()]=sha(path)
    dump(root/'source_manifest.json',source)
    protocol=dict(version=config['version'],frozen_utc=utc(),configuration=config,subject_partition=split,
        condition_names=CONDITION_NAMES,primary='engineered: D minus B in final test macro-F1 across 20 paired initialization/order seeds on one fixed subject partition',
        primary_interval='two-sided Student t 95% on paired seed differences, df=19; conditional on fixed subjects',
        secondary='Accuracy for primary pair; D-A, S08-A, S25-A, D-C, S25-D, S08-D in both representations; temporal D-B; losses, gaps, per-class/per-subject metrics and training costs. Unadjusted exploratory intervals.',
        additional_subject_interval='Average D-B accuracy/F1 differences across model seeds for each of the nine test subjects, then Student t interval over those nine subject means, df=8; descriptive, only nine subjects.',
        selection='Each condition and representation gets the same three SGD LRs and five tuning seeds. Maximize mean final validation macro-F1, ties lower mean validation cross-entropy then lower LR. Final fixed-budget checkpoint only. No retraining on validation subjects.',
        feature_layout='engineered j/560 in official metadata column order (semantic prior, not physical geometry); temporal time-major nine channels, t/127 shared by simultaneous channels. No values or labels define coordinates; no layout search.',
        preprocessing='Train-subject-only per-feature engineered mean/std; train-subject-only per-channel temporal mean/std across training windows and time. Same frozen scaler for all models/seeds. Vendor upstream normalization/filtering documented as a limitation.',
        fixed_locality='Both strict d50 .08 and medium .25 transferred as predeclared separate conditions from synthetic studies. Neither is chosen by test outcomes.',
        masks='B frozen independent within-row column permutations on both gated layers; C frozen permutation of input coordinates only; hidden gate matches D. D/B/C gate row-energy matched at every schedule point.',
        architecture='two ReLU hidden layers of 64, six-class dense output head, same learned parameters within representation. Dense A bypasses gate multiplication. All use multiclass cross-entropy and the original plain SGD optimizer.',
        batches='Each training epoch gets an independent permutation from order seed; precomputed stream shared across all conditions/LRs within seed. Fixed batch size 128 and 3200 updates for every model.',
        streams=dict(initialization=1000003,batches=2000003,masks=3000003,execution_order=4000003,subject_split=104729),
        test_embargo='Audit reads schema/counts, but tuning loads no test arrays and performs no model predictions on test. Freeze selected LRs before final models load test; evaluate native final test once. Subject/class metrics reuse those predictions. No test-time gate surgery or redesign.',
        timing='Single Torch thread per worker; six concurrent CPU workers. Record update wall and process CPU time excluding diagnostics, plus job elapsed wall. Execution order randomized independently within seed. Wall times under contention are not isolated latency benchmarks; no sparse-kernel speed claim.',
        subject_dependence='Overlapping windows stay within subjects. Seed intervals measure optimization variation on this fixed partition, not independent subject samples. Report subject-level results separately.',
        dataset_audit_sha256=sha(root/'audit/dataset_audit.json'),prepared_manifest_sha256=sha(root/'prepared_manifest.json'),
        source_manifest_sha256=sha(root/'source_manifest.json'),tuning_models=180,final_models=240)
    seal(root/'protocol.frozen.json',protocol)
    (root/'config.resolved.yaml').write_text(yaml.safe_dump(protocol,sort_keys=False),encoding='utf-8')
    (root/'config_sha256.txt').write_text(sha(root/'config.resolved.yaml')+'\n')
    git=subprocess.run(['git','rev-parse','HEAD'],cwd=REPO,capture_output=True,text=True)
    dump(root/'environment.json',dict(created_utc=utc(),python=sys.version,executable=sys.executable,
        platform=platform.platform(),torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,
        runtime_flags=flags,workers=args.workers,device='cpu',command=sys.argv,
        thread_environment={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']},
        git_revision=git.stdout.strip() if git.returncode==0 else None,
        source_revision=sha(root/'source_manifest.json')))
    packages=subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True,check=True)
    (root/'requirements.freeze.txt').write_text(packages.stdout,encoding='utf-8')
    print(f'PROTOCOL FROZEN: {root}',flush=True)
    tuning=run_phase(root,'tuning',config,{r:{c:config['learning_rates'] for c in config['conditions']} for r in config['representations']},args.workers)
    grid,selection=select_validation(tuning,config)
    grid.to_csv(root/'tuning/validation_grid.csv',index=False)
    seal(root/'selection.frozen.json',dict(frozen_utc=utc(),learning_rates=selection,
        tuning_scores_sha256=sha(root/'tuning/scores.csv'),protocol_sha256=sha(root/'protocol.frozen.json'),
        final_test_model_evaluated=False,source_manifest_sha256=sha(root/'source_manifest.json')))
    dump(root/'final_evaluation.started.json',dict(started_utc=utc(),selection_sha256=sha(root/'selection.frozen.json')))
    print(f'SELECTION FROZEN: {selection}',flush=True)
    final=run_phase(root,'replication',config,{r:{c:[lr] for c,lr in selection[r].items()} for r in selection},args.workers)
    from conn_exp.experiment2_report import build_report
    build_report(root)
    before=json.loads((root/'prior_artifacts.before.json').read_text())
    after={name:tree_manifest(REPO/'results/runs'/name) for name in before}
    dump(root/'prior_artifacts.after.json',after)
    assert before==after,'Historical artifacts changed'
    assert all(sha(REPO/name)==digest for name,digest in source.items()),'Source changed during experiment'
    assert all(sha(root/'prepared'/name)==digest for name,digest in prepared.items()),'Preprocessed data changed'
    for name in ['protocol.frozen.json','selection.frozen.json','subject_split.frozen.json']:
        assert sha(root/name)==(root/(name+'.sha256')).read_text().strip()
    dump(root/'completion.json',dict(completed_utc=utc(),tuning_models=len(tuning),final_models=len(final),
        historical_runs_unchanged=True,source_unchanged=True,prepared_data_unchanged=True,seals_verified=True))
    print(f'COMPLETE: {root}',flush=True)


if __name__=='__main__':
    main()
