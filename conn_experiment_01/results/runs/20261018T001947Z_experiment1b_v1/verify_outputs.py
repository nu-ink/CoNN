"""Independent final artifact checks; run with the project's Python interpreter."""
from pathlib import Path
import hashlib
import json
import math
import shutil
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest(folder):
    return {p.relative_to(folder).as_posix(): sha(p) for p in sorted(folder.rglob('*')) if p.is_file()}


def main():
    protocol = json.loads((ROOT / 'protocol.frozen.json').read_text())
    frozen = json.loads((ROOT / 'selection.frozen.json').read_text())
    started = json.loads((ROOT / 'replication.started.json').read_text())
    for name in ('protocol.frozen.json', 'selection.frozen.json'):
        assert sha(ROOT / name) == (ROOT / (name + '.sha256')).read_text().strip()
    assert frozen['frozen_utc'] < started['started_utc']
    assert started['selection_sha256'] == sha(ROOT / 'selection.frozen.json')
    assert frozen['tuning_scores_sha256'] == sha(ROOT / 'tuning/scores.csv')
    pilot = manifest(REPO / 'results/runs/20261017T235839Z_pilot')
    assert pilot == json.loads((ROOT / 'pilot_manifest.before.json').read_text())
    assert pilot == json.loads((ROOT / 'pilot_manifest.after.json').read_text())
    source = json.loads((ROOT / 'source_manifest.json').read_text())
    assert all(sha(ROOT / 'source' / name) == digest for name, digest in source.items())
    # Only descriptive reporting was extended after training, to explicitly discuss S.
    differences = [name for name, digest in source.items() if sha(REPO / name) != digest]
    assert differences == ['src/conn_exp/experiment1b_report.py']
    report_source = ROOT / 'postprocessing_source/experiment1b_report.py'
    report_source.parent.mkdir(exist_ok=True)
    shutil.copy2(REPO / differences[0], report_source)
    measurements = 0
    for phase, expected_count, expected_test_reads in [('tuning', 420, 0), ('replication', 140, 1)]:
        scores = pd.read_csv(ROOT / phase / 'scores.csv')
        assert len(scores) == expected_count
        assert (scores.test_evaluations == expected_test_reads).all()
        assert scores.groupby('seed').condition_id.nunique().eq(7).all()
        assert scores.groupby('seed').data_fingerprint.nunique().eq(1).all()
        assert scores.groupby('seed').init_fingerprint.nunique().eq(1).all()
        for row in scores.itertuples():
            job = ROOT / phase / row.job
            metrics = pd.read_csv(job / 'metrics.csv')
            test = metrics[metrics.split == 'test']
            assert len(test) == expected_test_reads
            if expected_test_reads:
                assert test.optimizer_steps_completed.eq(frozen['total_steps']).all()
                assert test.measure_timing.eq('post_step').all()
                assert row.lr == frozen['lr_by_condition'][row.condition_id]
            assert not metrics.non_finite.any()
            assert np.isfinite(metrics[['loss', 'accuracy']]).all().all()
            grads = pd.read_csv(job / 'gradients.csv')
            assert np.isfinite(grads[['mean_abs_dL_dW', 'mean_abs_dL_dW_eff']]).all().all()
            ablation = pd.read_csv(job / 'checkpoint_ablation.csv')
            assert len(ablation) == 9 and set(ablation.split) == {'val'}
            for checkpoint, subset in ablation.groupby('checkpoint'):
                assert subset.checkpoint_sha256.nunique() == 1
                assert sha(job / checkpoint) == subset.checkpoint_sha256.iloc[0]
            measurements += len(ablation)
    replication = pd.read_csv(ROOT / 'replication/scores.csv')
    pivot = replication.pivot(index='seed', columns='condition_id', values='test_accuracy')
    assert sorted(pivot.index) == protocol['replication_seeds']
    deltas = (pivot.D - pivot.B_prime).to_numpy()
    measured = json.loads((ROOT / 'replication/summary.json').read_text())
    mean, sd = float(deltas.mean()), float(deltas.std(ddof=1))
    half_width = 2.093024054408263 * sd / math.sqrt(20)
    for key, value in dict(mean=mean, sd=sd, ci_low=mean-half_width, ci_high=mean+half_width, cohen_dz=mean/sd).items():
        assert math.isclose(value, measured[key], abs_tol=1e-14)
    figures = list(ROOT.rglob('*.png'))
    assert len(figures) == 38
    for path in figures:
        with Image.open(path) as image:
            image.verify()
    result = dict(verified_utc=datetime.now(timezone.utc).isoformat(), pilot_unchanged=True,
        training_source_snapshot_verified=True, protocol_and_selection_seals_verified=True,
        freeze_precedes_replication=True, tuning_models=420, replication_models=140,
        tuning_test_evaluations=0, replication_test_evaluations=140,
        checkpoint_ablation_rows=measurements, pngs_verified=len(figures),
        all_metric_and_gradient_values_finite=True, primary_statistics_independently_recomputed=True,
        pre_training_test_suite='38 passed in 13.90s',
        post_training_reporting_change='Added all-condition mean table and explicit secondary S versus D interpretation; no training or selection changes.',
        postprocessing_source_sha256=sha(report_source), verifier_sha256=sha(Path(__file__)),
        frozen_training_source_manifest_sha256=sha(ROOT / 'source_manifest.json'))
    (ROOT / 'final_verification.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
