"""Measured post-training interpretation; no fitting, selection or test evaluation.

After regenerating experiment1c_report, run this script to restore its measured
mechanism findings. The frozen training/analysis source snapshot is unchanged.
"""
from pathlib import Path
import hashlib
import json
import re
from datetime import datetime, timezone
import pandas as pd

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    scores = pd.read_csv(ROOT/'replication/scores.csv')
    stats = pd.read_csv(ROOT/'replication/paired_statistical_summary.csv').set_index(['task','contrast'])
    core = scores[(scores.alignment_fraction==0)&~scores.shuffled]
    means = core.groupby(['task','policy'])[['train_accuracy','test_accuracy','test_loss']].mean()
    and_s = core[(core.task=='and')&(core.policy=='S')]
    below99 = int((and_s.train_accuracy<.99).sum())
    def effect(task,name):
        r = stats.loc[(task,name)]
        return f"{100*r['mean']:+.4f} points, 95% CI [{100*r.ci_low:+.4f}, {100*r.ci_high:+.4f}]"
    lines = ['## Measured mechanism findings', '',
        'The predeclared primary result is unresolved: its interval includes zero. The findings below are secondary or descriptive; the intervals are unadjusted. No settings were changed in response to these test results.', '',
        f"**The exact dense-release step is not a sufficient explanation.** AND W − D, with bitwise-identical pre-release trajectories, was {effect('and','W_minus_D')}. The immediate release loss seen in the old audit therefore does not establish a clear final benefit from removing that release. A short-lived gate-switch penalty and a final optimization outcome are different measurements.", '',
        f"**The earlier static-local ranking was seed-sensitive.** On these new seeds, AND S averaged {100*means.loc[('and','S'),'test_accuracy']:.4f}% versus D {100*means.loc[('and','D'),'test_accuracy']:.4f}%; S − D was {effect('and','S_minus_D')}. S did not retain a resolved superiority. Its mean training accuracy was {100*means.loc[('and','S'),'train_accuracy']:.4f}%, and {below99}/20 seeds ended below 99% training accuracy. This is consistent with variable fitting at the strict .08 local width within 3200 updates; it does not identify a specific gradient failure as the cause.", '',
        f"**Locality width and training history both matter in the observed AND ranking.** Fixed-medium F (d50=.25) had the highest observed core mean, {100*means.loc[('and','F'),'test_accuracy']:.4f}%; partial expansion P scored {100*means.loc[('and','P'),'test_accuracy']:.4f}%. P − F was {effect('and','P_minus_F_same_terminal_width')}. At the wider endpoint, W − H was {effect('and','W_minus_H_same_terminal_width')}. Thus the results do not support a general rule that expanding connectivity is harmful: the schedule comparison depends on its endpoint. F’s ranking is descriptive rather than a newly selected confirmatory winner.", '',
        f"**AND retains evidence consistent with task alignment.** S, P and D aligned-minus-row-shuffle differences were respectively {100*stats.loc[('and','S_aligned_minus_shuffle'),'mean']:+.4f}, {100*stats.loc[('and','P_aligned_minus_shuffle'),'mean']:+.4f} and {100*stats.loc[('and','D_aligned_minus_shuffle'),'mean']:+.4f} points. All three policies’ observed mean accuracy decreased as input-coordinate corruption increased from 0 to .5 to 1. The predeclared alignment-by-expansion interaction was {effect('and','alignment_by_expansion_interaction')}, so this sample does not resolve that interaction. These controls match row energy but do not isolate source alignment from all higher-order mask topology.", '',
        f"**The AND advantage did not generalize to the existing XOR task.** XOR D averaged {100*means.loc[('xor','D'),'test_accuracy']:.4f}% and its shuffled control averaged {100*scores[(scores.task=='xor')&(scores.variant_id=='D_shuffle_q0')].test_accuracy.mean():.4f}%. D minus its shuffle was {effect('xor','D_aligned_minus_shuffle')}. XOR P averaged {100*means.loc[('xor','P'),'test_accuracy']:.4f}% and S {100*means.loc[('xor','S'),'test_accuracy']:.4f}%. D and the dense baseline each reached 100% mean training accuracy, yet their test losses were {means.loc[('xor','D'),'test_loss']:.4f} and {means.loc[('xor','A'),'test_loss']:.4f}. This train/test mismatch is measured evidence of poor generalization under the fixed protocol, rather than success at learning XOR robustly. S also fitted the training data weakly ({100*means.loc[('xor','S'),'train_accuracy']:.4f}% mean accuracy). No XOR-driven redesign was performed.", '',
        f"**The historical D − B′ result remains intact.** Original 1B remains +4.843750 points. The new AND D-versus-shuffle secondary replication was {effect('and','D_aligned_minus_shuffle')}. These are separate seed samples and are not pooled.", '',
        'Together these measurements support a task- and policy-dependent role for locality on AND, with endpoint width and optimization history also relevant. They do not identify a universal dominant mechanism, prove a harmful collapse, or establish a geometry advantage across tasks. The primary partial-versus-dense comparison and AND alignment-by-expansion interaction remain unresolved at the declared precision.', '',
        'Post-training interpretation provenance: this section is generated by interpret_results.py from the saved scores and predeclared contrasts. To regenerate the entire report, run conn_exp.experiment1c_report first, then this script. Both stages use existing measurements only.', '']
    findings = '\n'.join(lines)
    (ROOT/'measured_findings.md').write_text(findings,encoding='utf-8')
    report = ROOT/'research_report.md'
    text = report.read_text(encoding='utf-8')
    text = re.sub(r'\n<!-- measured-findings:start -->.*?<!-- measured-findings:end -->\n', '', text, flags=re.S)
    base_digest = hashlib.sha256(text.encode()).hexdigest()
    insertion = '\n<!-- measured-findings:start -->\n'+findings+'<!-- measured-findings:end -->\n'
    text = text.replace('\n## Phase 1:', insertion+'\n## Phase 1:', 1)
    report.write_text(text,encoding='utf-8')
    # Compact aggregate diagnostics supplement the job-level raw records.
    metrics = pd.read_csv(ROOT/'replication/training_curves.csv')
    quantities = ['mean_abs_W_eff_ih','mean_abs_W_eff_hh','norm_W_eff_ih','norm_W_eff_hh',
        'gate_mean_square_ih','gate_mean_square_hh','row_energy_std_ih','row_energy_std_hh','act_var_h1','act_var_h2']
    metrics[(metrics.split=='val')&metrics.kind.isin(['initial','pre_release','post_release','final'])].groupby(
        ['task','variant_id','kind'])[quantities].agg(['mean','std']).to_csv(ROOT/'replication/weight_diagnostic_summary.csv')
    provenance = dict(created_utc=datetime.now(timezone.utc).isoformat(), stage='post-training descriptive interpretation',
        script_sha256=sha(Path(__file__)), scores_sha256=sha(ROOT/'replication/scores.csv'),
        contrasts_sha256=sha(ROOT/'replication/paired_statistical_summary.csv'),
        input_report_without_addendum_text_sha256=base_digest, output_report_sha256=sha(report),
        frozen_training_source_unchanged=True, new_models=0, new_test_evaluations=0)
    (ROOT/'interpretation.provenance.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
    print('Measured findings and research report updated.')


if __name__=='__main__':
    main()
