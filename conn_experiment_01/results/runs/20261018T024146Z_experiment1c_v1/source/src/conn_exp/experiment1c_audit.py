"""Read-only audit of Experiment 1B; no fitting and no new test evaluations."""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from conn_exp.experiment1b import REPO, dump, sha, tree_manifest, utc
from conn_exp.models import CoNNMLP, fingerprint_state
from conn_exp.runtime import configure_runtime
from conn_exp.train import evaluate_split

PRIOR = REPO / 'results/runs/20261018T001947Z_experiment1b_v1'
SNAPSHOTS = ['gate_pre_release.pt', 'gate_post_release.pt', 'final.pt']


def audit(root):
    root.mkdir(parents=True, exist_ok=False)
    folder = root / 'audit'
    folder.mkdir()
    prior_roots = [p for p in sorted((REPO / 'results/runs').iterdir()) if p.is_dir() and p != root]
    dump(root / 'prior_artifacts.before.json', {p.name: tree_manifest(p) for p in prior_roots})
    configure_runtime()
    protocol = json.loads((PRIOR / 'protocol.frozen.json').read_text())
    selection = json.loads((PRIOR / 'selection.frozen.json').read_text())
    # Read the published diagnostic tables explicitly; record precisely which versions.
    inputs = ['protocol.frozen.json', 'selection.frozen.json', 'research_summary.md',
        'replication/diagnostic_summary.csv', 'replication/checkpoint_ablation_summary.csv',
        'replication/summary.json', 'replication/plots/learning_curves.png']
    dump(folder / 'inputs.json', {p: sha(PRIOR / p) for p in inputs})
    diagnostic = pd.read_csv(PRIOR / 'replication/diagnostic_summary.csv', header=[0, 1], index_col=0)
    ablation = pd.read_csv(PRIOR / 'replication/checkpoint_ablation_summary.csv', header=[0, 1], index_col=[0, 1])
    diagnostic.to_csv(folder / 'existing_diagnostic_summary.csv')
    ablation.to_csv(folder / 'existing_checkpoint_ablation_summary.csv')
    scores = pd.read_csv(PRIOR / 'replication/scores.csv')
    release, total = round(.7*selection['total_steps']), selection['total_steps']
    boundaries, curve_frames, counterfactuals, checkpoint_checks = [], [], [], []
    for row in scores.itertuples():
        job = PRIOR / 'replication' / row.job
        metrics = pd.read_csv(job / 'metrics.csv')
        curve_frames.append(metrics)
        for label, completed, timing in [('pre_release', release, 'post_step'),
                ('post_release', release, 'pre_step'), ('final', total, 'post_step')]:
            frame = metrics[(metrics.optimizer_steps_completed == completed) & (metrics.measure_timing == timing)]
            for r in frame.itertuples():
                boundaries.append(dict(seed=row.seed, condition_id=row.condition_id, snapshot=label,
                    completed=completed, split=r.split, loss=r.loss, accuracy=r.accuracy,
                    origin='existing_1b_measurement'))
        if row.condition_id not in ['D', 'S']:
            continue
        data = np.load(PRIOR / 'replication/data' / f'seed_{row.seed}.npz')
        path = job / 'checkpoints' / f'seed_{row.seed}' / row.condition_id
        states = [torch.load(path / name, map_location='cpu', weights_only=False) for name in SNAPSHOTS]
        assert all(torch.equal(v, states[1]['model'][k]) for k, v in states[0]['model'].items())
        checkpoint_checks.append(dict(seed=row.seed, condition_id=row.condition_id,
            release_weights_identical=True, pre_completed=states[0]['optimizer_steps_completed'],
            post_completed=states[1]['optimizer_steps_completed']))
        for name, state, label in zip(SNAPSHOTS, states, ['pre_release', 'post_release', 'final']):
            model = CoNNMLP(100, (32, 32))
            model.load_state_dict(state['model'])
            before = fingerprint_state(model.state_dict())
            for split in ['train', 'val']:
                x, y = torch.from_numpy(data[f'x_{split}']), torch.from_numpy(data[f'y_{split}'])
                for gate_mode in ['native', 'exact_dense']:
                    ih, hh = state['gate_ih'], state['gate_hh']
                    if gate_mode == 'exact_dense':
                        ih, hh = torch.ones_like(ih), torch.ones_like(hh)
                    loss, acc = evaluate_split(model, x, y, ih, hh)
                    if gate_mode == 'native':
                        reference = [v for v in boundaries if v['seed'] == row.seed and v['condition_id'] == row.condition_id
                            and v['snapshot'] == label and v['split'] == split][0]
                        assert abs(loss-reference['loss']) < 2e-6 and acc == reference['accuracy']
                    counterfactuals.append(dict(seed=row.seed, condition_id=row.condition_id, snapshot=label,
                        split=split, gate_mode=gate_mode, loss=loss, accuracy=acc,
                        optimizer_steps_completed=state['optimizer_steps_completed'],
                        checkpoint=str((path/name).relative_to(PRIOR)), checkpoint_sha256=sha(path/name),
                        weight_fingerprint=before, origin='new_1c_inference_on_existing_1b_weights'))
            assert fingerprint_state(model.state_dict()) == before
    boundary = pd.DataFrame(boundaries)
    boundary.to_csv(folder / 'existing_boundary_measurements.csv', index=False)
    cf = pd.DataFrame(counterfactuals)
    cf.to_csv(folder / 'checkpoint_native_dense.csv', index=False)
    pd.DataFrame(checkpoint_checks).to_csv(folder / 'release_weight_checks.csv', index=False)
    boundary.groupby(['condition_id', 'snapshot', 'split'])[['loss', 'accuracy']].agg(['mean', 'std']).to_csv(folder / 'boundary_summary.csv')
    cf.groupby(['condition_id', 'snapshot', 'split', 'gate_mode'])[['loss', 'accuracy']].agg(['mean', 'std']).to_csv(folder / 'native_dense_summary.csv')
    changes = []
    for (seed, cid, split), group in boundary[boundary.split != 'test'].groupby(['seed', 'condition_id', 'split']):
        g = group.set_index('snapshot')
        for metric in ['loss', 'accuracy']:
            for effect, a, b in [('gate_switch_only', 'post_release', 'pre_release'),
                    ('further_optimization_960_steps', 'final', 'post_release')]:
                changes.append(dict(seed=seed, condition_id=cid, split=split, metric=metric, effect=effect,
                    delta=float(g.loc[a, metric]-g.loc[b, metric])))
    changes = pd.DataFrame(changes)
    changes.to_csv(folder / 'separated_changes.csv', index=False)
    change_summary = changes.groupby(['condition_id', 'split', 'metric', 'effect']).delta.agg(['mean', 'std'])
    change_summary.to_csv(folder / 'separated_change_summary.csv')
    curves = pd.concat(curve_frames)
    subset = curves[(curves.split == 'val') & ((curves.measure_timing == 'post_step') | (curves.optimizer_steps_completed == 0))]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for cid in ['D', 'S', 'B_prime']:
        for axis, metric in zip(axes, ['accuracy', 'loss']):
            g = subset[subset.condition_id == cid].groupby('optimizer_steps_completed')[metric].mean()
            axis.plot(g.index, g, label=cid)
            axis.axvline(release, color='gray', ls='--', lw=.7)
            axis.set_xlabel('optimizer steps completed')
            axis.set_ylabel('validation '+metric)
    axes[1].legend()
    fig.suptitle('Existing 1B replication curves; dashed line = 2240-step release')
    fig.tight_layout(); fig.savefig(folder / 'existing_learning_curves.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for cid in ['D', 'S']:
        for mode, style in [('native', '-'), ('exact_dense', '--')]:
            frame = cf[(cf.condition_id == cid) & (cf.gate_mode == mode) & (cf.split == 'val')]
            for metric, axis in zip(['accuracy', 'loss'], axes):
                avg = frame.groupby('snapshot')[metric].mean().reindex(['pre_release', 'post_release', 'final'])
                axis.plot(range(3), avg, style, marker='o', label=f'{cid} {mode}')
                axis.set_xticks(range(3), ['pre release', 'post release', 'final'])
                axis.set_ylabel('validation '+metric)
    axes[1].legend(fontsize=8)
    fig.suptitle('Existing weights, freshly measured gate interventions; no optimization')
    fig.tight_layout(); fig.savefig(folder / 'native_dense_checkpoints.png', dpi=150); plt.close(fig)
    lines = ['# Experiment 1C phase 1: audit before new training', '',
        'Historical 1B D − B′ replication remains +0.04843750, independently measured on seeds 100–119. No original artifacts or scores are overwritten, pooled, or reclassified.', '',
        'Read: frozen protocol and selection, diagnostic and checkpoint-ablation summaries, raw curves and saved checkpoints. Existing measurements are labeled separately from newly performed inference on existing weights. No new model was trained in this audit. New gate interventions use training and validation only; no new test evaluation is performed.', '',
        '## Existing validation checkpoints (20 seeds)', '',
        '| Condition | Snapshot | Accuracy | Loss |', '|---|---|---|---|']
    for cid in ['D', 'S', 'B_prime']:
        for label in ['pre_release', 'post_release', 'final']:
            frame = boundary[(boundary.condition_id == cid) & (boundary.snapshot == label) & (boundary.split == 'val')]
            lines.append(f'| {cid} | {label} | {frame.accuracy.mean():.6f} | {frame.loss.mean():.6f} |')
    lines += ['', '## Gate switch versus subsequent optimization', '']
    for cid in ['D', 'S', 'B_prime']:
        for effect in ['gate_switch_only', 'further_optimization_960_steps']:
            acc = change_summary.loc[(cid, 'val', 'accuracy', effect), 'mean']
            loss = change_summary.loc[(cid, 'val', 'loss', effect), 'mean']
            lines.append(f'- {cid}, {effect}: mean validation accuracy change {acc:+.8f}; mean loss change {loss:+.8f}.')
    lines += ['', 'The pre/post release snapshots contain bitwise-identical weights at 2240 updates. Their difference is a gate intervention. Post-release to final spans 960 actual optimization updates under a fixed terminal gate. S has no release gate change; its boundary difference is zero. These quantities must not be merged into one “collapse effect.”', '',
        '## Native versus dense gates on the same weights', '',
        '| Condition | Snapshot | Native val accuracy | Dense val accuracy | Native val loss | Dense val loss |', '|---|---|---|---|---|---|']
    for cid in ['D', 'S']:
        for label in ['pre_release', 'post_release', 'final']:
            f = cf[(cf.condition_id == cid) & (cf.snapshot == label) & (cf.split == 'val')].groupby('gate_mode')[['accuracy', 'loss']].mean()
            lines.append(f"| {cid} | {label} | {f.loc['native','accuracy']:.6f} | {f.loc['exact_dense','accuracy']:.6f} | {f.loc['native','loss']:.6f} | {f.loc['exact_dense','loss']:.6f} |")
    lines += ['', '## Diagnostic context and missing measurements', '',
        'The saved 1B final joint far-ablation validation loss deltas were D +0.07288950, B′ +0.88018548, S 0.0. S already has near-zero gates at far base distances, so zero additional effect does not mean that widening is intrinsically harmful. Effective weight magnitude differs substantially under S’s native gates. The copied diagnostic tables contain all layer norms and activation variances.',
        'Missing in 1B: held-out test metrics before release, test scores under counterfactual dense gates, pre-2240 saved weights beyond the three retained snapshots, and a branch trained after release while retaining the wide Gaussian instead of switching to exact dense. These are not reconstructed. Existing curve points are measurements, not interpolated checkpoint states. Counterfactual dense inference is newly measured and is not a missing historical measurement.',
        'Static-local superiority is observational across training procedures. Same-weight gate interventions assess immediate sensitivity, not the outcome of training with that gate. Whether persistent locality, gradual widening, exact release, or subsequent adaptation explains the difference requires the controlled follow-up.', '',
        'All phase-1 analyses are exploratory because the 1B outcomes motivated them. The 1B primary result is preserved as historical independent evidence, not reused as independent 1C replication.', '']
    (folder / 'diagnostic_report.md').write_text('\n'.join(lines), encoding='utf-8')
    dump(folder / 'completed.json', dict(completed_utc=utc(), new_models_trained=0, new_test_evaluations=0,
        checkpoint_inference_rows=len(cf), release_pairs_identical=len(checkpoint_checks),
        report_sha256=sha(folder / 'diagnostic_report.md'), input_protocol_sha256=sha(PRIOR / 'protocol.frozen.json')))
    print(f'AUDIT COMPLETE: {root}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path)
    args = parser.parse_args()
    root = args.run_dir or REPO / 'results/runs' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_experiment1c_v1')
    audit(root)
