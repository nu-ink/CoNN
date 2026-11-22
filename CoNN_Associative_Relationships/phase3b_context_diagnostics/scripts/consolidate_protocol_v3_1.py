"""Create the approved prospective protocol; never execute training or alter old artifacts."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REVIEW = ROOT / 'protocol_reviews/20261020_reference_reconciliation'
DEST = ROOT / 'protocol_versions/phase3b_v3_1_20261020'
RUN = ROOT / 'results/20261020_phase3b_v3_run01'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def immutable_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f'Refusing to replace existing versioned artifact: {path}')
    else:
        path.write_bytes(data)


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f'Expected exactly one substitution: {old[:100]}')
    return text.replace(old, new)


DECISIONS = {
    'A1': '''A1 decision: APPROVED for a future, versioned Phase 3B protocol.
For Gaussian Level 2 functional differentiation, use the pooled empirical shuffled-code null across all 35 Gaussian shuffled fits.
Calculate the NumPy linear 95th percentile separately for each layer and context pair (0,1), (0,2), and (1,2).
Apply identical live-support masks, normalization, numerical precision, and matrix-difference definitions to observed and null measurements.
Retain the absolute engineering threshold of 1e−4 and the same-weight functional-output requirement.
A Gaussian Level 2 claim requires the observed differentiation to exceed both the engineering threshold and its corresponding pooled shuffled-code threshold, together with the prescribed functional-output evidence.
Report all layer/pair comparisons, not only those exceeding thresholds. Do not select the best-performing pair after inspection.
Do not make Dense Level 2 null claims without separately calibrated Dense shuffled-code controls.
Label these as exploratory development diagnostics, not independent statistical significance.
If the current protocol was already frozen or results have been examined, record this as a prospective amendment rather than retroactively changing the decision rule.''',
    'A2': '''A2 decision: APPROVED — prospective Phase 3B Level 2 correction.
For future layer-specific Level 2 claims, require all of the following:
1. At least one context code in the evaluated pair exhibits a maximum absolute trained-versus-initial forward C change of ≥1e−6 on the pair's common live-support entries.
2. The trained pairwise normalized Frobenius difference exceeds both 1e−4 and the corresponding A1 Gaussian pooled shuffled-code 95th-percentile threshold.
3. Conduct layer-isolated functional interventions:
  - IH layer: Vary only the IH context code while keeping the HH context code fixed at 0. Keep weights and sensor input unchanged.
  - HH layer: Cache the same first hidden representation h1 and vary only the HH context code. Keep all downstream weights and other inputs unchanged.
4. The corresponding isolated intervention must produce an output difference above a numerical inference tolerance established and frozen before evaluating trained-model effects.
5. Evaluate all three context pairs and report every result, including failures.
6. Use identical input samples, live-support definitions, tensor precision, and normalization across the observed and shuffled-code comparisons.
7. Do not claim Dense Level 2 success using the Gaussian shuffled-code null.
8. Preserve earlier checkpoints, historical classifications, metrics, and conclusions without retroactive changes.
Record this amendment in a new versioned protocol before applying it to future experimental runs.''',
    'A3': '''A3 decision: APPROVED — prospective C3 futility rule.
Determine C3 futility separately for Dense and Gaussian architectures.
For each architecture, compute the maximum absolute trained-versus-initial C change on valid live-support entries, using the actual forward-pass dtype, across all required pilot checkpoints:
- Every 160 optimizer steps
- Pre-release
- Post-release
- Final
Use the complete checkpoint set rather than the final checkpoint alone.
If the maximum change is below 1e−6, omit only the remaining C3 full-grid fits for that architecture.
If the maximum change is at least 1e−6, that architecture's C3 condition remains eligible for full-grid execution.
Require a complete and verified pilot trajectory before making the decision. Missing or corrupted checkpoints must not be interpreted as zero modulation change.
Record the Dense and Gaussian decisions independently, including maximum changes, checkpoint identifiers, omitted fit counts, and reasons.
Do not replace omitted fits or change the planned conditions. Report planned, completed, failed, and omitted fit totals separately.
Passing the futility screen establishes only a minimum forward-modulation change; it does not establish Level 2 functional differentiation or a predictive benefit.
Apply this rule prospectively under a versioned protocol amendment. Preserve all prior runs and conclusions.''',
}


FUNCTIONAL = '''## 7. Live support and functional criteria — approved A1/A2

IH live support contains source coordinates potentially nonzero under the true mask; HH has all 64 potentially active sources. Compare each of (0,1), (0,2), (1,2), separately for IH and HH. Each pair uses the intersection of the two live supports, with nonintersections reported separately. IH (1,2) has empty intersection and is undefined: report this explicitly without assigning a zero, pass or failure. Report every fit/layer/pair, including failures and undefined comparisons; do not select a favorable pair or invent a study-wide success fraction.

For pair (a,b), layer L and common support S, define reference attenuation A_c=1-C_c using float64 context-master calculations. Define D_Lab=||A_a[:,S]-A_b[:,S]||F / max(||A_a[:,S]||F,||A_b[:,S]||F,float64 tiny). Apply exactly the same formula, masks, precision and matrix definitions to observed Gaussian C4 fits and Gaussian shuffled fits. Separately report attenuation actually used in float32 inference, reference/forward differences, singular values computed in float64, numerical rank and sigma2/sigma1. Rounded forward attenuation can have a different numerical rank; a reference rank is not proof of functional rank.

**A1 pooled null:** For each nonempty layer/pair, require all 35 completed and verified Gaussian shuffled full-grid fits (five seeds x seven folds). Compute Q_Lab=np.quantile(D_null,.95,method='linear'), pooling those 35 fits separately for that layer/pair. There is no seed/fold-specific null percentile and no averaging over pairs. Do not drop nonfinite/missing fits, substitute pilots, bootstrap a replacement null, or declare a null-based Level 2 result when the required null is incomplete. Identical fold-specific probe row identifiers are used for observed and shuffled models within each fold. CV overlap and reused development data make this an exploratory diagnostic threshold, not independent statistical significance. No Dense Level 2 null claim is permitted without separately calibrated Dense shuffled controls; such controls are outside this condition matrix.

**A2 pair-specific learning:** Save C_initial_forward and C_final_forward at the actual float32 effective-weight boundary. For each c in {a,b}, calculate M_Lc=max_{i,j in S}|C_final_forward(c)_ij-C_initial_forward(c)_ij|. Require max(M_La,M_Lb)>=1e-6 in the evaluated layer on that pair's common support. A change in another layer, another context, a nonintersecting entry, or the float64 reference alone cannot satisfy this requirement. Compute subtraction/reductions in float64 after obtaining the actual forward-dtype values. Record initial and final pair differences so random initial differentiation is not presented as learned variation.

**A2 isolated output intervention:** Retain the historical deterministic diagnostic probe rows: first 128 validation rows in the preserved fold-array order, with identical row identifiers across observed and shuffled models. Use final frozen weights and final scheduled gates; no optimization or outcome-based row selection. For IH pairs, apply their common sensor-support mask to the same probe once: (0,1) uses accel-only inputs, (0,2) gyro-only inputs; (1,2) is undefined because the intersection is empty. Vary only the IH code a versus b, keeping HH code fixed at 0 and input samples/weights unchanged. For HH, compute and cache one h1 from the same full-sensor probe with IH code 0, then vary only HH code a versus b, keeping h1 and all downstream weights fixed. Record logit RMS, max absolute difference, margins and prediction changes. These are same-representation interventions, not realistic availability predictions. The frozen output criterion is max_abs_logit_difference>1e-6 in each corresponding isolated intervention; compute reductions in float64 over actual float32 output tensors.

The 1e-6 inference floor is retained from the previous pre-training engineering specification, and is frozen here before any new trained-model effects. Before a future run, verify identical-state inference/reload determinism and the applicable CPU/CUDA numerical checks on untrained fixtures; do not raise or lower the tolerance after inspecting trained-model effects. A failed numerical check stops the affected stage and requires a documented revision, not a retrospective threshold adjustment.

**Level 2 conjunction:** An eligible Gaussian C4 layer/pair passes only if pair-specific learning meets >=1e-6, D_Lab>1e-4, D_Lab>Q_Lab, and its layer-isolated max output difference is >1e-6. Preserve each component result and all failures. Report descriptive Dense diagnostics without a Gaussian-null Level 2 classification. No separate p-value or universal representation-learning standard is implied.

Level 1 separately requires finite analytically consistent gradients, at least one actual stored context update, live forward C change >=1e-6, and valid C bounds. Level 3 separately requires primary mean>=+.005 and descriptive CI lower>0; attribution specifically to learned differentiation also requires valid Level 2 evidence. Passing C3 futility or Level 1 alone implies neither Level 2 nor prediction benefit.

Ordinary validation uses realistic sensor masks; interventions are separately labeled. Disable-C ablation sets C=1 without changing weights. Distinguish gate-switch effects from subsequent optimization. Do not treat full-sensor Phase 2 and mixed-mask Phase 3B as matched curricula.

'''


FUTILITY = '''For each architecture's C3 pilot, require the complete verified 3200-update trajectory and initial reference before making any futility decision. Persist all required checkpoint states rather than relying on an overwritten latest checkpoint. Required records are every 160 completed optimizer updates (160,320,...,3200), pre-release and post-release at 2240 completed updates with identical weights and schedule arguments 2239 and 2240 respectively, and the final checkpoint after 3200 updates. Record periodic/release/final identifiers separately even when update counts coincide; final schedule argument is 3199. Verify hashes, completed-update counts, initial/final replay and every required checkpoint. Missing, corrupt, nonfinite or unverified records stop the decision; they are never assigned zero change or automatically omitted.

For each verified checkpoint t, layer L and context c, compute max absolute C_t_forward(c)-C_initial_forward(c) on that context's valid live-support entries. C values are those actually used at the float32 effective-weight boundary; reductions may use float64. Define M_arch as the maximum over the entire required checkpoint set, both context layers and all three contexts. Independently for Dense and Gaussian: if M_arch<1e-6, omit that architecture's remaining 35 C3 full-grid fits; if M_arch>=1e-6, C3 remains eligible for its full grid. Record actual master update representability separately, but it is not an additional prerequisite for applying approved A3. Exactly 1e-6 is eligible. Save each architecture's maximum, argmax checkpoint/layer/context, complete checkpoint identifier/hash list, integrity outcome, omitted-fit identifiers/count and reason. Do not replace omitted fits or extrapolate unrecorded transient maxima. Passing this screen establishes only minimum forward modulation, not Level 2 or predictive benefit.'''


RESOLUTION = '''## 12. Reconciliation, approved decisions and freeze scope

V3 overrides the v2 fourteen-condition budget with fifteen conditions including frozen C4; retains C2/C4 U=-4.5; supplies operational Level 1/2/3 thresholds, all five permutations, independent streams, exact deterministic replay, percentile subject bootstrap and the three-member secondary family. The >=+.005 Level 3 rule supersedes v2 wording 'exceeds'. No historical artifact is amended.

| Decision | Prospective operative definition | Status |
|---|---|---|
| A1 | Pool all 35 Gaussian shuffled full-grid fits by layer/pair; NumPy linear 95th percentile; same masks/normalization/precision; no Dense null claim | User approved 2026-10-20 |
| A2 | Pair-common-support learning >=1e-6 in the evaluated layer for at least one code; IH-only with HH=0, HH-only with cached h1; isolated output >1e-6 | User approved 2026-10-20 |
| A3 | Complete verified pilot checkpoint set; maximum recorded live forward change; independent architecture decisions; <1e-6 omitted and >=1e-6 eligible | User approved 2026-10-20 |

The exact supplied A1–A3 decisions are archived in USER_DECISIONS_A1_A3.json. Explicit decisions take precedence over conflicting historical implementation wording. In particular A3 no longer conditions omission on an additional master-update prerequisite, and missing checkpoints never establish futility. The unchanged complete v2 reference, original v3 instructions and earlier review are hashed inputs; reference snapshots accompany this protocol.

The deterministic pilot, stream offsets, constant code, normalization, forward boundary, rank/finite-difference/inference thresholds and wall cap are retained explicitly from the historical implementation to make this continuity design reproducible. They are not a new validation-driven search. Existing limitations, including historical preprocessing contribution and reused engineering seeds/development data, remain visible. There are no unresolved A1–A3 definitions in this revision. No study-wide Level 2 aggregation criterion is invented.

This is protocol version phase3b_v3_1, prospective for future runs only, finalized after earlier results were known. It cannot be described as independently preregistered confirmation or applied retrospectively to reclassify the completed run. Earlier classifications, metrics, checkpoints and conclusions remain unchanged. Freeze the document and companion protocol_config.json together; verify their hashes and input provenance in PROTOCOL_SEAL.json. A protocol freeze is not an execution-ready claim: the historical trainer does not yet implement the corrected layer-specific criterion and persistent complete pilot checkpoints. Before any future performance training, a versioned implementation must pass numerical, masking, support, checkpoint-integrity and reproducibility gates, match this configuration, seal source/environment/data manifests, and record a fresh measured execution budget/backend. Do not invoke the old root launcher as though it implemented this amendment. No new fits or trained-model reanalyses are performed by this protocol-finalization task.
'''


def main():
    protected = {
        'historical_root_protocol': ROOT / 'PROTOCOL_PHASE3B_v3.md',
        'historical_run_protocol': RUN / 'PROTOCOL_PHASE3B_v3.md',
        'historical_completion': RUN / 'completion.json',
        'historical_delivery_manifest': RUN / 'delivery_manifest.json',
        'historical_configuration': ROOT / 'config/experiment.yaml',
        'earlier_review': REVIEW / 'PROTOCOL_PHASE3B_v3.md',
        'earlier_review_manifest': REVIEW / 'REVIEW_MANIFEST.json',
        'complete_v2_reference': ROOT / 'PROTOCOL_PHASE3B_v2_reference.md',
        'original_v3_amendment': RUN / 'references/user_phase3b_v3.txt',
    }
    before = {k: digest(p) for k, p in protected.items()}
    assert before['historical_root_protocol'] == '901e57c4d5dcc9bf63b05eb297c75948698db232eec15361248f8c791157e72c'
    assert before['complete_v2_reference'] == '844e8167bdf91cca4258c656b36a4372ea3886ca73068b74c25acdae430dcd0e'
    text = protected['earlier_review'].read_text(encoding='utf-8')
    first = text.index('## 1.')
    text = '''# Phase 3B v3.1 — consolidated prospective protocol

Status: FINALIZED PROSPECTIVE PROTOCOL; execution verification remains required.
Version: phase3b_v3_1. Design date: 2026-10-20 America/Chicago.
This incorporates the complete historical v2 reference, the original v3 amendment, and the explicit user-approved A1–A3 decisions. It preserves every earlier frozen protocol and run. It is prospective for future versioned experiments, after historical outcomes were known; no retrospective preregistration is claimed.

''' + text[first:]
    text = text.replace('This review uses the actual reference file', 'This protocol uses the actual reference file')
    text = text.replace('This review occurs after that run', 'This amendment occurs after that run')
    text = text.replace('not measurements newly generated by this review', 'not measurements newly generated by this amendment')
    text = replace_once(text, 'Retaining them is proposed for continuity;', 'This protocol retains them for continuity;')
    text = replace_once(text, 'The existing harness used seed400/fold0 first paired batch and step0 gates.',
                        'Use seed400/fold0 first paired training batch and step0 gates, retaining the existing deterministic diagnostic fixture.')
    begin, end = text.index('## 7.'), text.index('## 8.')
    text = text[:begin] + FUNCTIONAL + text[end:]
    text = replace_once(text, 'For continuity, the existing deterministic pilot plan is retained as a proposal, not claimed to be explicitly specified by v2:',
                        'The frozen deterministic pilot plan retains the existing implementation choice, which was not explicitly specified by v2:')
    begin = text.index("After each architecture's complete C3 pilot")
    end = text.index('\n\nV3 replaces', begin)
    text = text[:begin] + FUTILITY + text[end:]
    text = replace_once(text, 'as the proposed continuity cap', 'as the frozen continuity cap')
    text = replace_once(text, 'No new training starts during this review.', 'No new training starts during this protocol-finalization task.')
    text = replace_once(text, 'An unresolved draft is not an operative protocol seal.',
                        'A protocol seal does not waive implementation or execution verification gates.')
    text = text[:text.index('## 12.')] + RESOLUTION
    config = {
        'version': 'phase3b_v3_1', 'prospective_only': True, 'execution_ready': False,
        'steps': 3200, 'learning_rate': .1, 'batch_size': 128,
        'optimizer': 'SGD', 'momentum': 0, 'weight_decay': 0, 'gradient_clipping': False,
        'seeds': list(range(400,405)),
        'folds': [[14,21,29],[6,7,17],[22,23,28],[8,11,26],[5,15,16],[3,25,30],[1,19,27]],
        'hidden_sizes': [64,64], 'factor_rank': 2, 'factor_seed': 30031,
        'factor_draw_dtype': 'float64', 'factor_draw_order': ['ih.u','ih.v','hh.u','hh.v'],
        'primary': 'gaussian_C4-minus-gaussian_masked',
        'conditions': [f'{a}_C{i}' for a in ('dense','gaussian') for i in range(5)] +
                      ['dense_masked','gaussian_masked','gaussian_constant','gaussian_shuffled','gaussian_frozen'],
        'pilot_conditions': [f'{a}_C{i}' for a in ('dense','gaussian') for i in range(5)],
        'pilot_seed': 400, 'pilot_fold': 0, 'pilot_steps': 3200,
        'maximum_full_grid_fits': 525, 'maximum_pilot_fits': 10,
        'maximum_logical_training_updates': 1712000, 'maximum_wall_seconds': 21600,
        'primary_minimum_effect': .005, 'primary_ci_lower_must_exceed': 0,
        'bootstrap_seed': 7000031, 'bootstrap_resamples': 10000, 'bootstrap_method': 'paired_subject_percentile_95',
        'level2': {
            'architectures_with_calibrated_null': ['gaussian'],
            'observed_condition': 'gaussian_C4', 'null_condition': 'gaussian_shuffled',
            'null_fit_count': 35, 'null_pool_axes': ['seed','fold'],
            'null_stratify_axes': ['layer','context_pair'], 'null_quantile': .95,
            'numpy_quantile_method': 'linear', 'null_must_be_complete': True,
            'context_pairs': [[0,1],[0,2],[1,2]], 'layers': ['ih','hh'],
            'support': 'pair_intersection', 'empty_support': 'undefined_reported',
            'matrix_dtype': 'float64_reference_attenuation', 'forward_dtype': 'float32',
            'normalization': 'frobenius_difference_divided_by_max_pair_attenuation_norm_float64_tiny',
            'engineering_floor_strictly_exceed': .0001,
            'pair_layer_learning_delta_at_least': .000001,
            'learning_rule': 'at_least_one_code_in_pair_on_same_layer_common_support',
            'isolated_logit_max_abs_strictly_exceed': .000001,
            'probe_rows': 'first_128_validation_rows_in_preserved_fold_order',
            'ih_downstream_hh_code': 0, 'ih_probe_mask': {'0,1':1,'0,2':2,'1,2':None},
            'hh_cached_h1_input_mask': 0, 'hh_cached_h1_ih_code': 0,
            'all_pairs_reported': True, 'dense_null_claims': False,
        },
        'c3_futility': {
            'separate_architectures': True, 'threshold': .000001,
            'omit_if': 'maximum_over_complete_required_checkpoints_strictly_below_threshold',
            'eligible_if': 'maximum_at_least_threshold', 'requires_complete_verified_pilot': True,
            'missing_checkpoint_action': 'stop_no_futility_decision',
            'master_update_is_additional_prerequisite': False,
            'forward_dtype': 'float32', 'periodic_completed_updates': list(range(160,3201,160)),
            'release_completed_updates': 2240, 'release_schedule_arguments': {'pre':2239,'post':2240},
            'final_completed_updates': 3200, 'final_schedule_argument': 3199,
            'persist_all_required_states': True, 'include_final_and_release_identifiers': True,
            'omitted_grid_fits_per_architecture': 35, 'replacement_fits': 0,
        },
        'official_test_evaluations': 0, 'phase4_permitted': False,
    }
    assert len(config['conditions']) == 15 and len(config['pilot_conditions']) == 10
    assert (525+10)*3200 == config['maximum_logical_training_updates']
    assert 'unresolved below' not in text and 'NOT FROZEN' not in text
    assert 'same seed/fold/layer/pair shuffled-code model null' not in text
    decisions = {'source': 'Explicit user replies to A1–A3 on 2026-10-20 America/Chicago',
                 'scope': 'prospective future versioned experiments only', 'decisions': DECISIONS}
    outputs = {
        'PROTOCOL_PHASE3B_v3.md': text.encode('utf-8'),
        'protocol_config.json': (json.dumps(config,indent=2)+'\n').encode('utf-8'),
        'USER_DECISIONS_A1_A3.json': (json.dumps(decisions,indent=2,ensure_ascii=False)+'\n').encode('utf-8'),
    }
    for name, data in outputs.items():
        immutable_write(DEST/name, data)
    for key in ('complete_v2_reference','original_v3_amendment'):
        immutable_write(DEST/'references'/protected[key].name, protected[key].read_bytes())
    immutable_write(DEST/'references/prior_review_manifest.json', (REVIEW/'REVIEW_MANIFEST.json').read_bytes())
    after = {k: digest(p) for k,p in protected.items()}
    if before != after:
        raise RuntimeError('Protected input changed during protocol generation')
    # Verification/sealing is a separate command, so no draft can be accidentally called frozen.
    print(json.dumps({'status':'generated_for_final_verification', 'directory':str(DEST),
                      'protected_inputs_unchanged':True, 'training_fits':0},indent=2))


if __name__ == '__main__':
    main()
