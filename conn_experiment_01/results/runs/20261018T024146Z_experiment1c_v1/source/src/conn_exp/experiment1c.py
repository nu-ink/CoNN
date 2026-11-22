"""Frozen, paired locality-versus-collapse study. Run the read-only audit first."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
import yaml

from conn_exp.config import load_config
from conn_exp.data import assert_dataset_invariants, generate_dataset
from conn_exp.experiment1b import REPO, dump, seal, sha, tree_manifest, utc
from conn_exp.experiment1c_core import POLICIES, PolicyGates, Variant, alignment_permutation, train_variant, variants
from conn_exp.gates import draw_masks
from conn_exp.geometry import make_task_geometry
from conn_exp.models import initial_state_dict
from conn_exp.runner import _save_splits
from conn_exp.runtime import configure_runtime
from conn_exp.train import minibatch_order

TUNING_SEEDS = tuple(range(10, 15))
REPLICATION_SEEDS = tuple(range(200, 220))


def select_shared_lr(scores, seeds=TUNING_SEEDS):
    """Each task uses one LR for all policies, preserving common pre-release paths."""
    columns = ['task', 'variant_id', 'lr', 'seed', 'alignment_fraction', 'val_accuracy', 'val_loss']
    frame = scores[columns].copy()  # No test measurement is an input.
    if frame.duplicated(['task', 'variant_id', 'lr', 'seed']).any():
        raise ValueError('Duplicate tuning candidate')
    if not np.isfinite(frame[['val_accuracy', 'val_loss']]).all().all():
        raise ValueError('Nonfinite tuning measurement')
    for task in ['and', 'xor']:
        expected = {(task, v.id, rate, seed) for v in variants(task) for rate in [.01, .05, .1] for seed in seeds}
        actual = set(frame[frame.task == task][['task', 'variant_id', 'lr', 'seed']].itertuples(index=False, name=None))
        if actual != expected:
            raise ValueError('Incomplete or unequal tuning budgets')
    core = frame[frame.alignment_fraction == 0]
    grid = core.groupby(['task', 'lr'], as_index=False).agg(
        macro_val_accuracy=('val_accuracy', 'mean'), macro_val_loss=('val_loss', 'mean'), n=('seed', 'size'))
    selected = grid.sort_values(['task', 'macro_val_accuracy', 'macro_val_loss', 'lr'],
        ascending=[True, False, True, True]).drop_duplicates('task')
    return grid, {r.task: float(r.lr) for r in selected.itertuples()}


def worker_batch(root_string, phase, task, seed, rates):
    root = Path(root_string) / phase
    configure_runtime()
    base = load_config(REPO/'configs/pilot.yaml').override(task=task, total_steps=3200,
        eval_interval=160, seed_list=(seed,))
    dataset = generate_dataset(base, seed)
    assert_dataset_invariants(dataset, base)
    data_folder = root/'data'/task
    data_folder.mkdir(parents=True, exist_ok=True)
    if phase == 'replication':
        _save_splits(data_folder/f'seed_{seed}.npz', dataset)
    else:
        np.savez_compressed(data_folder/f'seed_{seed}.npz', x_train=dataset.train.x,
            y_train=dataset.train.y, x_val=dataset.val.x, y_val=dataset.val.y)
    masks = draw_masks(base.n_features, *base.hidden_sizes, seed)
    init = initial_state_dict(base.n_features, base.hidden_sizes, seed)
    order = minibatch_order(base.n_train, seed)
    geometry = make_task_geometry(base.n_features, base.hidden_sizes)
    paired = root/'paired_inputs'/task
    paired.mkdir(parents=True, exist_ok=True)
    torch.save(dict(initial_state=init, masks=vars(masks), order=order,
        alignment_permutations={str(q): alignment_permutation(100, seed, q) for q in [0., .5, 1.]}),
        paired/f'seed_{seed}.pt')
    rows = []
    journal = root/'journals'/f'{task}_seed_{seed}.jsonl'
    for rate in rates:
        config = base.override(lr=rate)
        for variant in variants(task):
            job = root/'jobs'/task/f'seed_{seed}'/f'lr_{rate:g}'/variant.id
            score, _ = train_variant(config, variant, seed, init, masks, geometry, order,
                dataset, job, replication=phase == 'replication')
            score['job'] = job.relative_to(root).as_posix()
            with journal.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(score, allow_nan=False)+'\n')
            rows.append(score)
            print(f'{phase} {task} seed={seed} {variant.id} lr={rate:g} done', flush=True)
    return rows


def audit_phase(root, scores):
    """Validate all pairings, energy matches and common-path checkpoints."""
    assert scores.parameter_count.nunique() == 1 and scores.parameter_count.iloc[0] == 4321
    for _, group in scores.groupby(['task', 'seed']):
        for column in ['data_fingerprint', 'init_fingerprint', 'order_sha256', 'total_steps']:
            assert group[column].nunique() == 1, column
    common_path = scores[(scores.alignment_fraction == 0) & scores.policy.isin(['W', 'D'])]
    for _, group in common_path.groupby(['task', 'seed', 'lr', 'shuffled']):
        assert len(group) == 2 and group.pre_release_weight_fingerprint.nunique() == 1
    max_energy_gap = 0.
    for (task, seed, rate), group in scores.groupby(['task', 'seed', 'lr']):
        for policy in POLICIES:
            rows = group[(group.policy == policy) & (group.alignment_fraction == 0)]
            a = pd.read_csv(root/rows[~rows.shuffled].iloc[0].job/'row_energy.csv')
            b = pd.read_csv(root/rows[rows.shuffled].iloc[0].job/'row_energy.csv')
            match = a.merge(b, on=['optimizer_steps_completed', 'kind', 'layer', 'unit'], suffixes=('_a', '_b'))
            gap = float((match.energy_a-match.energy_b).abs().max())
            max_energy_gap = max(max_energy_gap, gap)
            assert len(match) == len(a) == len(b) and gap < 1e-3
        for row in group[group.alignment_fraction > 0].itertuples():
            reference = group[(group.policy == row.policy) & ~group.shuffled & (group.alignment_fraction == 0)].iloc[0]
            a = pd.read_csv(root/row.job/'row_energy.csv')
            b = pd.read_csv(root/reference.job/'row_energy.csv')
            match = a.merge(b, on=['optimizer_steps_completed', 'kind', 'layer', 'unit'], suffixes=('_a', '_b'))
            assert len(match) == len(a) == len(b)
            gap = float((match.energy_a-match.energy_b).abs().max())
            max_energy_gap = max(max_energy_gap, gap)
            assert gap < 1e-3
    result = dict(paired_data_initialization_order=True, parameter_count=4321,
        all_budgets_3200_steps=bool(scores.total_steps.eq(3200).all()),
        common_W_D_pre_release_weights_identical=True, per_unit_gate_energy_max_abs_gap=max_energy_gap,
        test_evaluations=int(scores.test_evaluations.sum()))
    dump(root/'invariant_audit.json', result)


def execute_phase(root, phase, seeds, rates, workers):
    phase_root = root/phase
    phase_root.mkdir(exist_ok=False)
    (phase_root/'journals').mkdir()
    scores = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(worker_batch, str(root), phase, task, seed, rates[task])
            for seed in seeds for task in ['and', 'xor']]
        for future in as_completed(futures):
            scores.extend(future.result())
            pd.DataFrame(scores).to_csv(phase_root/'scores.csv', index=False)
            print(f'BATCH COMPLETE {phase}: {len(scores)} models saved to aggregate', flush=True)
    frame = pd.DataFrame(scores).sort_values(['task', 'seed', 'variant_id', 'lr'])
    frame.to_csv(phase_root/'scores.csv', index=False)
    audit_phase(phase_root, frame)
    return frame


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True, help='Completed audit root from experiment1c_audit')
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args(argv)
    root = args.run_dir.resolve()
    if not (root/'audit/completed.json').is_file():
        raise RuntimeError('Complete phase-1 audit before any new training')
    if not 1 <= args.workers <= (os.cpu_count() or 1):
        raise ValueError('Invalid worker budget')
    flags = configure_runtime()
    source = {}
    files = [p for folder in ['src', 'tests', 'configs'] for p in (REPO/folder).rglob('*')
        if p.is_file() and '__pycache__' not in p.parts]
    files += [REPO/'pyproject.toml', REPO/'README.md', REPO/'DESIGN_DECISIONS.md']
    for path in sorted(files):
        relative = path.relative_to(REPO)
        target = root/'source'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        source[relative.as_posix()] = sha(path)
    dump(root/'source_manifest.json', source)
    base = load_config(REPO/'configs/pilot.yaml')
    protocol = dict(version='1C-v1', frozen_utc=utc(), audit_sha256=sha(root/'audit/diagnostic_report.md'),
        base_config=base.to_resolved_dict(), total_steps=3200, release_step=2240,
        policies={p: asdict(v) for p, v in POLICIES.items()}, dense_baseline='A exact dense at every step',
        variants={task: [dict(id=v.id, **asdict(v)) for v in variants(task)] for task in ['and', 'xor']},
        tuning_seeds=list(TUNING_SEEDS), replication_seeds=list(REPLICATION_SEEDS),
        learning_rate_grid=[.01, .05, .1], tuning_models=480, replication_models=640,
        selection='One common LR per task: maximize mean final validation accuracy over all 13 q=0 core variants and five tuning seeds; ties lower mean validation loss then lower LR. Every degradation variant also gets the full equal grid but does not enter selection. Freeze task LRs before replication.',
        primary='AND: P_aligned_q0 minus D_aligned_q0, native final test accuracy on 20 new paired seeds',
        primary_interval='two-sided 95% Student t on paired seed differences; df=19; sole primary contrast',
        secondary=['W-D: isolate exact dense release, common Gaussian prehistory and LR',
            'S-D: fixed locality versus full release', 'P-F and W-H: schedule versus static at matched terminal Gaussian width',
            'aligned minus row-shuffled for all six terminal policies',
            '(P-P_shuffle)-(S-S_shuffle): alignment-by-expansion interaction',
            'D-B_prime analogue is a new secondary replication; historical 1B remains unchanged',
            'XOR: repeat all core policy and alignment contrasts',
            'AND degradation q=.5 and q=1 versus q=0 for S,P,D, and difference versus dense baseline'],
        secondary_inference='descriptive, exploratory, unadjusted 95% intervals; no secondary confirmatory claims',
        alignment_degradation='Only input coordinates permuted; hidden geometry and all data columns/labels unchanged. Independently ordered prefix of 50 or 100 coordinates cyclically permuted; exactly 50%/100% moved; affected sets nested, mappings need not be. Row-wise gate multisets and energy preserved. q is corruption fraction, not physical displacement.',
        random_streams=dict(data=0, initialization=1000003, order=2000003, row_masks=3000003, alignment=4000003),
        test_usage='No tuning test evaluation. Replication native final test exactly once. Gate comparisons and far ablations use validation only. No test-selected redesign, checkpoints, LRs or replacement seeds.',
        checkpoints=['pre_release: 2240 updates, previous native gate', 'post_release: same weights, next native gate', 'final: 3200 updates'],
        diagnostics='Every 160 updates plus same-weight release pair and final. Full train/val curves; effective weights; per-unit gate energy; fixed first-minibatch dL/dW and dL/dW_eff by base geometry bins; validation far ablation in ih, hh and jointly. Serialized checkpoint reloads under native/dense gates.',
        convergence='First measured validation accuracy >=.8,.9,.95; sampling interval 160 updates; missing threshold remains missing; no early stopping',
        matched_parameters=4321, optimizer='original SGD momentum=0 weight_decay=0 foreach=False fused=False',
        preservation='Before/after SHA256 manifest of every pre-existing run, including 1A/pilot and 1B; no historical results pooled',
        source_manifest_sha256=sha(root/'source_manifest.json'))
    seal(root/'protocol.frozen.json', protocol)
    (root/'config.resolved.yaml').write_text(yaml.safe_dump(protocol, sort_keys=False), encoding='utf-8')
    (root/'config_sha256.txt').write_text(sha(root/'config.resolved.yaml')+'\n')
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, capture_output=True, text=True)
    dump(root/'environment.json', dict(created_utc=utc(), python=sys.version, executable=sys.executable,
        platform=platform.platform(), torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__,
        device='cpu', runtime_flags=flags, worker_processes=args.workers,
        threads={key: os.environ.get(key) for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']},
        code_revision=git.stdout.strip() if git.returncode == 0 else 'content-addressed source snapshot; no Git repository',
        source_manifest_sha256=sha(root/'source_manifest.json'), command=sys.argv))
    packages = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True, check=True)
    (root/'requirements.freeze.txt').write_text(packages.stdout, encoding='utf-8')
    print(f'PROTOCOL FROZEN: {root}', flush=True)
    tuning = execute_phase(root, 'tuning', TUNING_SEEDS, {'and': [.01,.05,.1], 'xor': [.01,.05,.1]}, args.workers)
    grid, rates = select_shared_lr(tuning)
    grid.to_csv(root/'tuning/shared_lr_validation.csv', index=False)
    seal(root/'selection.frozen.json', dict(frozen_utc=utc(), learning_rate_by_task=rates,
        tuning_scores_sha256=sha(root/'tuning/scores.csv'), protocol_sha256=sha(root/'protocol.frozen.json'),
        source_manifest_sha256=sha(root/'source_manifest.json'), primary=protocol['primary'],
        replication_seeds=list(REPLICATION_SEEDS), replication_test_evaluated=False))
    dump(root/'replication.started.json', dict(started_utc=utc(), selection_sha256=sha(root/'selection.frozen.json')))
    print(f'SELECTION FROZEN BEFORE REPLICATION: {rates}', flush=True)
    replication = execute_phase(root, 'replication', REPLICATION_SEEDS, {task: [rate] for task, rate in rates.items()}, args.workers)
    from conn_exp.experiment1c_report import build_report
    build_report(root)
    before = json.loads((root/'prior_artifacts.before.json').read_text())
    after = {name: tree_manifest(REPO/'results/runs'/name) for name in before}
    dump(root/'prior_artifacts.after.json', after)
    assert before == after, 'Historical artifacts changed'
    assert all(sha(REPO/path) == digest for path, digest in source.items()), 'Source changed during study'
    for name in ['protocol.frozen.json', 'selection.frozen.json']:
        assert sha(root/name) == (root/(name+'.sha256')).read_text().strip()
    dump(root/'completion.json', dict(completed_utc=utc(), tuning_models=len(tuning), replication_models=len(replication),
        prior_artifacts_unchanged=True, frozen_source_unchanged=True, seals_verified=True))
    print(f'COMPLETE: {root}', flush=True)


if __name__ == '__main__':
    main()
