"""Paired seed statistics; one primary contrast, six Holm-adjusted secondary tests."""
from __future__ import annotations
import argparse
from itertools import product
import json
from pathlib import Path
import numpy as np
from scipy import stats
from .core import Config, CONDITIONS, TASKS
from .runner import write_json


def paired(values, alternative='two-sided'):
    d = np.asarray(values, dtype=float)
    if len(d) < 2 or not np.isfinite(d).all():
        raise ValueError('at least two finite paired replications required')
    n = len(d)
    mean, sd = float(d.mean()), float(d.std(ddof=1))
    half = float(stats.t.ppf(.975, n-1) * sd / np.sqrt(n))
    if n > 20:
        raise ValueError('exact sign-flip enumeration limited to 20 seeds')
    flips = np.array(list(product((-1, 1), repeat=n)), dtype=float) @ d / n
    if alternative == 'greater':
        p = float(np.mean(flips >= mean - 1e-12))
    elif alternative == 'two-sided':
        p = float(np.mean(np.abs(flips) >= abs(mean) - 1e-12))
    else:
        raise ValueError('unknown alternative')
    return {'n': n, 'differences': d.tolist(), 'mean_difference': mean, 'sd_difference': sd,
            'ci95_t': [mean-half, mean+half], 'cohen_dz': mean/sd if sd else None,
            'exact_signflip_p': p, 'alternative': alternative,
            'paired_t_p_sensitivity': float(stats.ttest_1samp(d, 0, alternative=alternative).pvalue) if sd else None}


def holm(pvalues):
    order = np.argsort(pvalues)
    adjusted = np.zeros(len(order))
    bound = 0.0
    for i, idx in enumerate(order):
        bound = max(bound, (len(order)-i)*pvalues[idx])
        adjusted[idx] = min(1.0, bound)
    return adjusted.tolist()


def analyze(run):
    run = Path(run)
    cfg = Config.load(run / 'config.json')
    rows = json.loads((run / 'evaluation' / 'metrics.json').read_text())
    expected = len(cfg.seeds) * len(TASKS) * len(CONDITIONS) * 3
    if len(rows) != expected:
        raise ValueError('incomplete metric table')
    lookup = {(r['task'], r['endpoint'], r['condition'], r['seed']): r for r in rows}
    if len(lookup) != expected:
        raise ValueError('duplicate endpoint records')
    for task in TASKS:
        for seed in cfg.seeds:
            for ep in ('pre_release', 'immediate_release', 'final'):
                group = [lookup[task, ep, c, seed] for c in CONDITIONS]
                for k in ('test_sha256', 'initialization_sha256', 'completed_steps'):
                    if len({r[k] for r in group}) != 1:
                        raise ValueError(f'pairing mismatch: {k}')

    def diff(task='structured', endpoint='final', control='scrambled'):
        return np.array([lookup[task, endpoint, 'aligned', s]['accuracy'] - lookup[task, endpoint, control, s]['accuracy'] for s in cfg.seeds])

    primary = paired(diff(), 'greater')
    primary['practical_effect'] = cfg.practical_effect
    primary['statistically_positive'] = primary['exact_signflip_p'] < .05 and primary['mean_difference'] > 0
    primary['practical_support'] = primary['statistically_positive'] and primary['ci95_t'][0] > cfg.practical_effect
    secondary = {
        'structured_final_aligned_minus_mlp': paired(diff(control='mlp')),
        'structured_final_aligned_minus_uniform': paired(diff(control='uniform')),
        'structured_pre_release_aligned_minus_scrambled': paired(diff(endpoint='pre_release')),
        'structured_immediate_release_aligned_minus_scrambled': paired(diff(endpoint='immediate_release')),
        'negative_final_aligned_minus_scrambled': paired(diff(task='negative')),
        'structured_minus_negative_alignment_effect': paired(diff()-diff(task='negative')),
    }
    for r, p in zip(secondary.values(), holm([v['exact_signflip_p'] for v in secondary.values()])):
        r['holm_p'] = p
    summary = []
    for task in TASKS:
        for ep in ('pre_release', 'immediate_release', 'final'):
            for c in CONDITIONS:
                entries = [lookup[task, ep, c, s] for s in cfg.seeds]
                row = {'task': task, 'endpoint': ep, 'condition': c}
                for m in ('accuracy', 'f1', 'loss'):
                    row[f'{m}_mean'] = float(np.mean([e[m] for e in entries]))
                    row[f'{m}_sd'] = float(np.std([e[m] for e in entries], ddof=1))
                summary.append(row)
    runtime = []
    for task in TASKS:
        for c in CONDITIONS:
            entries = [json.loads((run / f'seed_{s}' / task / c / 'status.json').read_text()) for s in cfg.seeds]
            dense = [json.loads((run / f'seed_{s}' / task / 'mlp' / 'status.json').read_text()) for s in cfg.seeds]
            ratios = [r['optimization_seconds']/d['optimization_seconds'] for r, d in zip(entries, dense)]
            runtime.append({'task': task, 'condition': c,
                            'optimization_seconds_mean': float(np.mean([r['optimization_seconds'] for r in entries])),
                            'wall_seconds_mean': float(np.mean([r['wall_seconds'] for r in entries])),
                            'paired_optimization_ratio_mean': float(np.mean(ratios)),
                            'paired_optimization_ratios': ratios,
                            'sampled_rss_bytes_max': max(r['peak_sampled_process_rss_bytes'] for r in entries)})
    report = {'primary': primary, 'secondary': secondary, 'condition_summary': summary,
              'runtime': runtime, 'seeds': cfg.seeds, 'inference_unit': 'independent paired DGP and training seed',
              'interval_assumption': 'Student t interval for independent seed differences; symmetry assumed by sign-flip test',
              'secondary_intervals': 'unadjusted descriptive 95% intervals; Holm adjustment applies to six p-values',
              'other_metrics': 'F1/loss/curves/weight-distance/runtime/memory descriptive only'}
    path = run / 'analysis.json'
    if path.exists():
        raise FileExistsError(path)
    write_json(path, report)
    print(json.dumps({'primary': primary, 'secondary': secondary}, indent=2))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    analyze(p.parse_args().run)
