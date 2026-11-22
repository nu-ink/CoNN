"""Validate and seal the prospective design, without importing or running ML code."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
REVIEW = ROOT / 'protocol_reviews/20261020_reference_reconciliation'
RUN = ROOT / 'results/20261020_phase3b_v3_run01'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    protocol = HERE / 'PROTOCOL_PHASE3B_v3.md'
    text = protocol.read_text(encoding='utf-8')
    cfg = load(HERE/'protocol_config.json')
    decisions = load(HERE/'USER_DECISIONS_A1_A3.json')
    prior = load(REVIEW/'REVIEW_MANIFEST.json')
    historical = {name: Path(info['path']) for name, info in prior['inputs'].items()}
    historical['earlier_review'] = REVIEW/'PROTOCOL_PHASE3B_v3.md'
    historical['earlier_review_manifest'] = REVIEW/'REVIEW_MANIFEST.json'
    historical['historical_delivery_manifest'] = RUN/'delivery_manifest.json'
    protected_before = {name: sha(path) for name,path in historical.items()}
    rows = [line.split('|')[1].strip() for line in text.splitlines()
            if line.startswith('| dense_') or line.startswith('| gaussian_')]
    l2, fut = cfg['level2'], cfg['c3_futility']
    periodic = list(range(160,3201,160))
    checks = {
        'historical_selected_inputs_still_match_prior_review': all(
            protected_before[name] == info['sha256'] for name, info in prior['inputs'].items()),
        'old_protocol_seal_unchanged': load(RUN/'protocol_seal.json')['protocol_sha256'] ==
            protected_before['preserved_root_protocol'],
        'previous_review_preserved': protected_before['earlier_review'] == prior['draft_sha256'],
        'complete_v2_snapshot_identical': (HERE/'references/PROTOCOL_PHASE3B_v2_reference.md').read_bytes() ==
            historical['complete_v2_reference'].read_bytes(),
        'original_v3_snapshot_identical': (HERE/'references/user_phase3b_v3.txt').read_bytes() ==
            historical['original_v3_amendment'].read_bytes(),
        'three_explicit_approved_decisions_archived': set(decisions['decisions']) == {'A1','A2','A3'} and
            all('APPROVED' in v for v in decisions['decisions'].values()),
        'unique_matrix_matches_configuration': rows == cfg['conditions'] and len(set(rows)) == 15,
        'unchanged_common_optimizer_and_pairing': (cfg['optimizer'],cfg['learning_rate'],cfg['batch_size'],cfg['steps']) ==
            ('SGD',.1,128,3200) and cfg['seeds'] == list(range(400,405)) and len(cfg['folds']) == 7,
        'fifteen_conditions_budget': len(rows)*len(cfg['seeds'])*len(cfg['folds']) == cfg['maximum_full_grid_fits'] == 525,
        'ten_deterministic_pilots': cfg['pilot_conditions'] == rows[:10] and
            (cfg['pilot_seed'],cfg['pilot_fold'],cfg['pilot_steps']) == (400,0,3200),
        'training_budget_arithmetic': (cfg['maximum_full_grid_fits']+cfg['maximum_pilot_fits'])*cfg['steps'] ==
            cfg['maximum_logical_training_updates'] == 1712000,
        'explicit_wall_cap': cfg['maximum_wall_seconds'] == 21600 and '21600-second wall cap' in text,
        'A1_null_population_and_method': l2['null_fit_count'] == 35 and l2['null_pool_axes'] == ['seed','fold'] and
            l2['null_stratify_axes'] == ['layer','context_pair'] and l2['numpy_quantile_method'] == 'linear' and
            l2['null_quantile'] == .95 and l2['null_must_be_complete'] and "method='linear'" in text,
        'A1_no_dense_null_claims': not l2['dense_null_claims'] and l2['architectures_with_calibrated_null'] == ['gaussian'] and
            'No Dense Level 2 null claim is permitted' in text,
        'A1_A2_all_pairs_and_common_support': l2['context_pairs'] == [[0,1],[0,2],[1,2]] and
            l2['all_pairs_reported'] and l2['support'] == 'pair_intersection' and l2['empty_support'] == 'undefined_reported',
        'A2_pair_layer_learning_and_threshold': l2['learning_rule'] ==
            'at_least_one_code_in_pair_on_same_layer_common_support' and
            l2['pair_layer_learning_delta_at_least'] == 1e-6 and 'max(M_La,M_Lb)>=1e-6' in text,
        'A2_engineering_and_isolated_output_floors': l2['engineering_floor_strictly_exceed'] == 1e-4 and
            l2['isolated_logit_max_abs_strictly_exceed'] == 1e-6 and
            'max_abs_logit_difference>1e-6' in text,
        'A2_isolated_input_and_hidden_probes': l2['ih_downstream_hh_code'] == 0 and
            l2['hh_cached_h1_ih_code'] == 0 and l2['hh_cached_h1_input_mask'] == 0 and
            l2['ih_probe_mask'] == {'0,1':1,'0,2':2,'1,2':None} and 'then vary only HH code' in text,
        'A2_observed_null_precision_and_rows': l2['matrix_dtype'] == 'float64_reference_attenuation' and
            l2['forward_dtype'] == 'float32' and l2['probe_rows'] == 'first_128_validation_rows_in_preserved_fold_order' and
            'same formula, masks, precision and matrix definitions' in text,
        'A3_complete_checkpoint_set': fut['periodic_completed_updates'] == periodic and
            fut['release_completed_updates'] == 2240 and fut['release_schedule_arguments'] == {'pre':2239,'post':2240} and
            fut['final_completed_updates'] == 3200 and fut['final_schedule_argument'] == 3199 and
            fut['persist_all_required_states'] and fut['include_final_and_release_identifiers'],
        'A3_missing_data_stops_decision': fut['requires_complete_verified_pilot'] and
            fut['missing_checkpoint_action'] == 'stop_no_futility_decision' and
            'never assigned zero change or automatically omitted' in text,
        'A3_architecture_and_boundary_rule': fut['separate_architectures'] and fut['threshold'] == 1e-6 and
            not fut['master_update_is_additional_prerequisite'] and fut['forward_dtype'] == 'float32' and
            'if M_arch<1e-6' in text and 'if M_arch>=1e-6' in text and
            fut['omitted_grid_fits_per_architecture'] == 35 and fut['replacement_fits'] == 0,
        'primary_and_bootstrap_preserved': cfg['primary'] == 'gaussian_C4-minus-gaussian_masked' and
            cfg['primary_minimum_effect'] == .005 and cfg['bootstrap_resamples'] == 10000,
        'no_unresolved_draft_rules': all(s not in text for s in (
            'NOT FROZEN FOR NEW TRAINING', 'unresolved below', 'A1–A3 remain unresolved',
            'as a proposal', 'as the proposed continuity cap', 'same seed/fold/layer/pair shuffled-code model null')),
        'prospective_not_execution_ready': cfg['prospective_only'] and not cfg['execution_ready'] and
            'the historical trainer does not yet implement' in text and
            'Earlier classifications, metrics, checkpoints and conclusions remain unchanged' in text,
        'official_test_and_phase4_embargo': cfg['official_test_evaluations'] == 0 and not cfg['phase4_permitted'],
    }
    if not all(checks.values()):
        raise RuntimeError(json.dumps({k:v for k,v in checks.items() if not v}, indent=2))
    # These assertions verify specification consistency; they are not model/unit-test results.
    protected_after = {name: sha(path) for name,path in historical.items()}
    checks['historical_inputs_unchanged_during_sealing'] = protected_after == protected_before
    if protected_after != protected_before:
        raise RuntimeError('Protected inputs changed during sealing')
    files = sorted(p for p in HERE.rglob('*') if p.is_file() and p.name not in
                   {'PROTOCOL_SEAL.json','PROTOCOL_PHASE3B_v3.sha256'})
    hashes = {str(p.relative_to(HERE)).replace('\\','/'):sha(p) for p in files}
    seal_path = HERE/'PROTOCOL_SEAL.json'
    if seal_path.exists():
        seal = load(seal_path)
        if seal['files'] != hashes or seal['protected_inputs'] != protected_before or seal['checks'] != checks:
            raise RuntimeError('Existing seal mismatch; do not overwrite. Create a versioned amendment.')
        state = 'existing_seal_verified'
    else:
        seal = {
            'version':'phase3b_v3_1', 'sealed_at_utc':datetime.now(timezone.utc).isoformat(),
            'design_date_timezone':'America/Chicago',
            'status':'prospective_protocol_frozen_execution_not_ready',
            'protocol_sha256':sha(protocol), 'configuration_sha256':sha(HERE/'protocol_config.json'),
            'files':hashes, 'protected_inputs':protected_before, 'checks':checks,
            'remaining_A1_A2_A3_ambiguities':[], 'new_training_fits':0,
            'new_optimizer_updates':0, 'trained_model_effects_reanalyzed':False,
            'official_test_evaluations':0, 'phase4_started':False,
            'execution_gate':'New versioned implementation and source/config/data/environment seals plus numerical/pilot gates required.',
            'scope_limit':'Document/configuration consistency and selected historical provenance; not a full artifact rehash or executed-model validation.',
        }
        seal_path.write_text(json.dumps(seal,indent=2)+'\n',encoding='utf-8')
        state = 'new_protocol_seal_created'
    checksum = sha(protocol)+'  PROTOCOL_PHASE3B_v3.md\n'
    checksum_path = HERE/'PROTOCOL_PHASE3B_v3.sha256'
    if checksum_path.exists() and checksum_path.read_text(encoding='ascii') != checksum:
        raise RuntimeError('Protocol checksum mismatch')
    if not checksum_path.exists():
        checksum_path.write_text(checksum,encoding='ascii')
    print(json.dumps({'status':state,'checks_passed':sum(checks.values()),'checks_total':len(checks),
                      'protocol_sha256':sha(protocol),'execution_ready':False,'new_training_fits':0},indent=2))


if __name__ == '__main__':
    main()
