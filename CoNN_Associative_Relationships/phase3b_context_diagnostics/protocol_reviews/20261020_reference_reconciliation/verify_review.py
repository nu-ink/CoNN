"""Read-only scientific inputs; write review evidence only. Does not import ML code."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RUN = ROOT / 'results/20261020_phase3b_v3_run01'
EXPECTED_PROTOCOL = '901e57c4d5dcc9bf63b05eb297c75948698db232eec15361248f8c791157e72c'
EXPECTED_REFERENCE = '844e8167bdf91cca4258c656b36a4372ea3886ca73068b74c25acdae430dcd0e'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    seal = json.loads((RUN / 'protocol_seal.json').read_text(encoding='utf-8'))
    completion = json.loads((RUN / 'completion.json').read_text(encoding='utf-8'))
    draft = HERE / 'PROTOCOL_PHASE3B_v3.md'
    text = draft.read_text(encoding='utf-8')
    actual = [line.split('|')[1].strip() for line in text.splitlines()
              if line.startswith('| dense_') or line.startswith('| gaussian_')]
    expected = [f'{arch}_C{i}' for arch in ('dense', 'gaussian') for i in range(5)]
    expected += ['dense_masked', 'gaussian_masked', 'gaussian_constant',
                 'gaussian_shuffled', 'gaussian_frozen']
    references = {
        'complete_v2_reference': ROOT / 'PROTOCOL_PHASE3B_v2_reference.md',
        'original_v3_amendment': RUN / 'references/user_phase3b_v3.txt',
        'preserved_root_protocol': ROOT / 'PROTOCOL_PHASE3B_v3.md',
        'preserved_run_protocol': RUN / 'PROTOCOL_PHASE3B_v3.md',
        'preserved_source_protocol': RUN / 'source/PROTOCOL_PHASE3B_v3.md',
        'preserved_configuration': ROOT / 'config/experiment.yaml',
        'preserved_completion': RUN / 'completion.json',
        'observed_reporting_implementation': ROOT / 'src/reporting.py',
        'observed_diagnostics_implementation': ROOT / 'src/diagnostics.py',
    }
    before = {name: sha(path) for name, path in references.items()}
    checks = {
        'reference_matches_fully_read_file': before['complete_v2_reference'] == EXPECTED_REFERENCE,
        'root_protocol_retains_frozen_hash': before['preserved_root_protocol'] == EXPECTED_PROTOCOL,
        'run_and_source_protocols_match_root': all(before[name] == EXPECTED_PROTOCOL for name in
                                                   ('preserved_run_protocol', 'preserved_source_protocol')),
        'seal_still_names_original_protocol': seal['protocol_sha256'] == EXPECTED_PROTOCOL,
        'completed_run_still_names_original_protocol': completion['protocol_seal']['protocol_sha256'] == EXPECTED_PROTOCOL,
        'matrix_exactly_fifteen_unique_cells': actual == expected and len(set(actual)) == 15,
        'budget_525_grid_535_total_1712000_steps': 15*7*5 == 525 and (15*7*5+10)*3200 == 1712000,
        'two_c3_omissions_arithmetic': (525-2*35+10)*3200 == 1488000,
        'three_open_ambiguities_explicit': all(f'| A{i} |' in text for i in (1, 2, 3)),
        'draft_explicitly_not_frozen': 'NOT FROZEN FOR NEW TRAINING' in text,
        'not_certified_conflict_free': 'NOT certified conflict-free' in text,
        'training_not_started_by_review': 'No new training starts during this review.' in text,
    }
    if not all(checks.values()):
        raise RuntimeError(json.dumps(checks, indent=2))
    snapshots = HERE / 'references'
    snapshots.mkdir(exist_ok=True)
    for name in ('complete_v2_reference', 'original_v3_amendment'):
        target = snapshots / references[name].name
        if target.exists() and target.read_bytes() != references[name].read_bytes():
            raise RuntimeError(f'Refusing to replace review snapshot: {target}')
        if not target.exists():
            target.write_bytes(references[name].read_bytes())
    after = {name: sha(path) for name, path in references.items()}
    checks['all_observed_historical_inputs_unchanged_during_verification'] = before == after
    if before != after:
        raise RuntimeError('Historical input changed during review verification')
    digest = sha(draft)
    manifest = {
        'reviewed_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'review_draft_unresolved_not_frozen',
        'content_checksum_is_not_protocol_seal': True,
        'new_training_fits': 0,
        'new_optimizer_updates': 0,
        'official_test_evaluations': 0,
        'phase4_started': False,
        'open_ambiguities': ['A1_null_population', 'A2_layer_attribution', 'A3_pilot_temporal_summary'],
        'checks': checks,
        'draft_sha256': digest,
        'verifier_sha256': sha(Path(__file__)),
        'inputs': {name: {'path': str(path), 'sha256': before[name]} for name, path in references.items()},
        'scope_limit': 'Document and selected provenance checks; not a full artifact rehash or semantic proof.',
    }
    (HERE / 'REVIEW_MANIFEST.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    (HERE / 'PROTOCOL_PHASE3B_v3.review.sha256').write_text(digest+'  PROTOCOL_PHASE3B_v3.md\n', encoding='ascii')
    print(json.dumps({'checks_passed': sum(checks.values()), 'checks_total': len(checks),
                      'draft_sha256': digest, 'training_ready': False}, indent=2))


if __name__ == '__main__':
    main()
