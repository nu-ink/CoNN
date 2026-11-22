"""Generate evidence-linked documentation and a standalone scientific figure from saved runs."""
from pathlib import Path
import csv
import json
import sys
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'experiments'/'experiment_04'
DOC=ROOT/'docs'/'research_validation'/'experiment_04'
RUNS={'smoke':'run_20261121T031643_767335Z_7b6dcbc2',
      'pilot':'run_20261121T031701_870795Z_ab99ae48',
      'confirmatory':'run_20261121T031743_424624Z_7ac51c80'}
RUN=BASE/'results'/RUNS['confirmatory']
analysis=json.loads((RUN/'analysis.json').read_text())
rows=json.loads((RUN/'evaluation'/'metrics.json').read_text())
seeds=analysis['seeds']
colors={'mlp':'#687986','aligned':'#00796b','scrambled':'#b44528','uniform':'#6650a2'}


def put(path,text):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f: f.write(text.strip()+'\n')


def mdtable(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)


def test_counts(path):
    root=ET.parse(path).getroot()
    suites=root.findall('testsuite') if root.tag=='testsuites' else [root]
    return {key:sum(int(s.attrib.get(key,0)) for s in suites) for key in ('tests','failures','errors','skipped')}


def main():
    DOC.mkdir(parents=True,exist_ok=True)
    # Export all endpoint metrics and exact paired primary differences.
    with (RUN/'endpoint_metrics.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (RUN/'primary_paired_differences.csv').open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['seed','aligned_accuracy','scrambled_accuracy','difference'])
        for s in seeds:
            a=next(r['accuracy'] for r in rows if (r['seed'],r['task'],r['condition'],r['endpoint'])==(s,'structured','aligned','final'))
            b=next(r['accuracy'] for r in rows if (r['seed'],r['task'],r['condition'],r['endpoint'])==(s,'structured','scrambled','final'))
            w.writerow([s,a,b,a-b])
    # Figure: independent seeds, paired outcomes, validation convergence and endpoints.
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    ax=axes[0,0]
    for s in seeds:
        a=[next(r['accuracy'] for r in rows if (r['seed'],r['task'],r['condition'],r['endpoint'])==(s,'structured',c,'final'))*100 for c in ('scrambled','aligned')]
        ax.plot([0,1],a,color='#9ca9ad',alpha=.65,linewidth=1)
        ax.scatter([0,1],a,color=[colors['scrambled'],colors['aligned']],s=22,zorder=3)
    ax.set(xticks=[0,1],xticklabels=['Scrambled','Aligned'],ylabel='Held-out accuracy (%)',title='Paired final accuracy: 10 independent seeds')
    ax=axes[0,1]
    p=analysis['primary'];vals=np.array(p['differences'])*100
    ax.axhline(0,color='black',linewidth=.7);ax.axhline(2,color='#9d6a00',linestyle='--',label='Practical threshold: 2 pp')
    ax.scatter(seeds,vals,color=colors['aligned'])
    ax.set(xlabel='Paired seed',ylabel='Aligned minus scrambled (pp)',title='Primary paired differences');ax.ticklabel_format(useOffset=False);ax.legend(fontsize=8)
    curve_summary=[]
    terminal=[]
    ax=axes[1,0]
    for task in ('structured','negative'):
        for c in colors:
            curves=[json.loads((RUN/f'seed_{s}'/task/c/'curves.json').read_text()) for s in seeds]
            for i in range(len(curves[0])):
                row={'task':task,'condition':c,'completed_steps':curves[0][i]['completed_steps']}
                for split in ('train','val'):
                    for m in ('accuracy','loss'):
                        row[f'{split}_{m}_mean']=float(np.mean([cr[i][split][m] for cr in curves]))
                curve_summary.append(row)
            terminal.append([task,c,f"{100*np.mean([cr[-1]['train']['accuracy'] for cr in curves]):.2f}",
                             f"{100*np.mean([cr[-1]['val']['accuracy'] for cr in curves]):.2f}",
                             f"{np.mean([cr[-1]['train']['loss'] for cr in curves]):.4f}",
                             f"{np.mean([cr[-1]['val']['loss'] for cr in curves]):.4f}"])
            if task=='structured':
                ax.plot([r['completed_steps'] for r in curves[0]],100*np.mean([[r['val']['accuracy'] for r in cr] for cr in curves],axis=0),label=c,color=colors[c])
    ax.axvline(1120,color='gray',linestyle='--',linewidth=.8)
    ax.set(xlabel='Completed updates',ylabel='Mean validation accuracy (%)',title='Structured validation: fixed budget, no selection');ax.legend(fontsize=8)
    ax=axes[1,1]
    for c in ('aligned','scrambled'):
        for task,style in (('structured','-'),('negative','--')):
            v=[next(r['accuracy_mean'] for r in analysis['condition_summary'] if (r['task'],r['condition'],r['endpoint'])==(task,c,e))*100 for e in ('pre_release','immediate_release','final')]
            ax.plot(range(3),v,marker='o',linestyle=style,color=colors[c],label=f'{task}: {c}')
    ax.set(xticks=range(3),xticklabels=['Active gate','Immediate release','Final dense'],ylabel='Mean test accuracy (%)',title='Fixed endpoints; no model selection');ax.legend(fontsize=8)
    fig.suptitle('CoNN Experiment 04 — Relationship Geometry Validation',fontsize=15)
    fig.savefig(RUN/'experiment04_summary.png',dpi=180)
    fig.savefig(RUN/'experiment04_summary.pdf')
    plt.close(fig)
    put(RUN/'convergence_summary.json',json.dumps(curve_summary,indent=2))
    put(RUN/'analysis_environment.json',json.dumps({'scipy':scipy.__version__,'matplotlib':matplotlib.__version__},indent=2))

    prefix='../../../experiments/experiment_04/'
    evidence=f"[Confirmatory run]({prefix}results/{RUN.name}/), [analysis JSON]({prefix}results/{RUN.name}/analysis.json), [raw endpoint table]({prefix}results/{RUN.name}/endpoint_metrics.csv)."
    finaltable=mdtable(['Task','Condition','Accuracy mean ± SD (%)','Binary F1 mean ± SD','BCE mean ± SD'],[
        [r['task'],r['condition'],f"{100*r['accuracy_mean']:.3f} ± {100*r['accuracy_sd']:.3f}",
         f"{r['f1_mean']:.4f} ± {r['f1_sd']:.4f}",f"{r['loss_mean']:.4f} ± {r['loss_sd']:.4f}"]
        for r in analysis['condition_summary'] if r['endpoint']=='final'])
    secondarytable=mdtable(['Contrast','Mean effect (pp)','95% t interval (pp), unadjusted','Exact two-sided p','Holm p'],[
        [name,f"{100*v['mean_difference']:.3f}",f"[{100*v['ci95_t'][0]:.3f}, {100*v['ci95_t'][1]:.3f}]",f"{v['exact_signflip_p']:.6f}",f"{v['holm_p']:.6f}"]
        for name,v in analysis['secondary'].items()])
    runtime=mdtable(['Task','Condition','Optimization seconds/fit','Mean paired ratio to mlp','Max sampled process RSS (MiB)'],[
        [r['task'],r['condition'],f"{r['optimization_seconds_mean']:.3f}",f"{r['paired_optimization_ratio_mean']:.3f}",f"{r['sampled_rss_bytes_max']/2**20:.1f}"] for r in analysis['runtime']])
    primarytable=mdtable(['Seed','Aligned − scrambled (pp)'],[[s,f'{100*d:.4f}'] for s,d in zip(seeds,p['differences'])])
    unit=test_counts(BASE/'results'/'unit_junit.xml');reg=test_counts(BASE/'results'/'regression_junit.xml')
    tests=mdtable(['Suite','Passed','Failures','Errors','Skipped'],[
        ['Experiment 04',unit['tests']-unit['failures']-unit['errors']-unit['skipped'],unit['failures'],unit['errors'],unit['skipped']],
        ['Existing relevant regressions',reg['tests']-reg['failures']-reg['errors']-reg['skipped'],reg['failures'],reg['errors'],reg['skipped']]])

    put(DOC/'EXPERIMENT_DESIGN.md',f'''# Experiment design

Status: executed. {evidence}

The complete [preregistration]({prefix}PREREGISTRATION.md) was written before the first fit.
It specifies the DGP, endpoints, 2 pp threshold, one-sided primary test, stopping rule,
six secondary contrasts and runtime criterion. Earlier Experiment 04 proposal files
remain historical proposals; this directory documents the authorized implementation.

One manipulation is source assignment of Gaussian gate values, preserving every
scrambled row's multiset. A layer-uniform control matches mean squared magnitude.
All four conditions share raw initial weights, 4,321 parameters, architecture,
SGD configuration, batches, data and 1,600 updates. Uniform matches layer energy,
whereas scrambled matches row energy. Effective initial weights differ by treatment.

Structured features have a known latent-interaction grouping. The negative control
permutes observed columns independent of targets and keeps coordinates fixed. Its
geometry is unrelated by construction, with chance residual alignment possible.
The target never references Gaussian gates. Hidden coordinates encode a prior,
not identified semantic neurons. No geometry is learned.

Smoke: one independent seed, eight fits and three saved evaluation endpoints per fit.
Pilot: two disjoint seeds, 16 fits; training and validation only, no test evaluation.
Confirmatory: ten new seeds (4000–4009), 80 fits; 240 fixed endpoint measurements.
Train/validation/test sizes are 512/256/2048 for pilot/main, 64/64/128 for smoke.
No validation-based hyperparameter change or model selection occurred.

The primary endpoint is final independent test accuracy after 480 dense updates.
Pre-release and immediate-release endpoints are secondary and share identical weights.
All main models were trained before the separate test-evaluation command ran.
Each seed independently samples its latent data, initialization and mask.

No Git checkout or commit exists; the user explicitly waived checkout after the
preflight blocker. Original files are protected by a hash manifest and source snapshots.
No packages were installed or upgraded. All work used the existing Python 3.10 environment.
''')
    put(DOC/'MATHEMATICAL_SPECIFICATION.md',f'''# Mathematical specification

Implementation: [core.py]({prefix}src/conn_exp04/core.py), with unmodified historical
[gates](../../../conn_experiment_01/src/conn_exp/gates.py),
[geometry](../../../conn_experiment_01/src/conn_exp/geometry.py),
[models](../../../conn_experiment_01/src/conn_exp/models.py) and
[schedule](../../../conn_experiment_01/src/conn_exp/schedules.py).

For sample n, g=0,...,4, independently draw a_ng,b_ng ~ N(0,1). For k=0,...,19,
x_n,(20g+k) = a_ng + sigma*epsilon if k<10, otherwise b_ng + sigma*epsilon,
with independent epsilon~N(0,1), sigma=.3.

S_n = sum_g (|a_ng+b_ng| - |a_ng-b_ng|), c_n=1[S_n>0],
y_n=c_n XOR Bernoulli(.05). Each pair contributes
2*sign(a_ng*b_ng)*min(|a_ng|,|b_ng|), an actual interaction between two latents.
Negating all a variables negates S without changing its sampling distribution;
P(c=1)=.5 except zero-probability ties. Observed split class proportions fluctuate.

Within a repeated-measurement subgroup, off-diagonal covariance is 1 and
variance is 1+sigma². Independent subgroups have zero population covariance.
The label additionally couples a and b within each group nonlinearly.
Negative task: x'_i=x_pi(i) with a label-independent permutation pi frozen per seed.
No preprocessing is fitted, and independent named split streams generate new examples.

Fixed source coordinate s_i=i/(n_source-1), destination d_j=j/(n_dest-1).
D_ji=|d_j-s_i|, shape [n_dest,n_source], so distance and tau share unit-interval units.
G_ji=exp(-D_ji²/(2*tau²)); G(0)=1 and dG/dD=-(D/tau²)G <= 0.
The half-magnitude distance is d50=tau*sqrt(2 ln 2).

For release R=1120 and t<R, u=t/(R-1), tau(t)=tau0+(tau1-tau0)*u²(3-2u),
tau0=.08/sqrt(2 ln 2), tau1=1/sqrt(2 ln 2).
At t>=R, replace all gates with exact ones, including the scrambled and uniform controls.
Aligned uses G; scrambled uses G_(j,pi_j(i)), with pi_j fixed independently per row;
uniform uses sqrt(mean(G²)) in every element; mlp always uses ones.
Thus scrambled sum_i G_ji² is identical mathematically; floating diagnostics accumulate
in float64, with exact sorted-value equality checked separately at every step.

W_eff=W*stopgrad(G). A forward pass is
h1=ReLU(W1_eff*x+b1), h2=ReLU(W2_eff*h1+b2), z=w_out*h2+b_out.
Loss is mean binary cross-entropy with logits. The parameter gradient is
dL/dW=G*(dL/dW_eff); gates, coordinates and tau are not parameters.
For constant G, effective SGD update is proportional to G²; this is a prescribed
optimization bias as well as an initial scaling intervention.

Count: 32*100+32+32*32+32+32+1=4321 trainable scalars.
No extra trainable state appears in any condition. Gate tensors remain dense.
Finite-difference fixture uses float64, epsilon=1e-6, atol=1e-9, rtol=1e-6.
''')
    put(DOC/'STATISTICAL_ANALYSIS.md',f'''# Statistical analysis

{evidence}

Independent unit: paired seed (n=10), not individual examples or minibatches.
The primary contrast was frozen before fitting and test access.
Aligned minus scrambled final structured test accuracy: mean **1.1328125 pp**,
sample SD **1.2771871 pp**, two-sided 95% t CI **[0.2191679, 2.0464571] pp**,
Cohen dz **0.8869589**. Exact one-sided sign-flip p=**0.0146484375**;
paired t-test sensitivity p=0.01027525595.

There are 1,024 possible sign assignments for ten differences. The exact test
enumerates them and compares their mean with the observed mean, including ties.
This relies on a symmetric null distribution of seed differences, stronger than
an unrestricted mean-zero null. The t interval assumes an approximately well-behaved
independent seed mean; ten replications cannot validate those assumptions strongly.
The preregistered test rejects at .05, but practical support required the lower
95% bound to exceed 2 pp. That criterion is **not met**. The point estimate is also
below 2 pp. This is not a demonstration of equivalence or absence of a useful effect.

{primarytable}

Six two-sided secondary sign-flip tests form one Holm-adjusted family:

{secondarytable}

Secondary intervals are descriptive and unadjusted. The adjusted evidence supports
the aligned-minus-MLP contrast and the structured-minus-negative interaction at .05.
The aligned-minus-uniform contrast does not survive adjustment; gate magnitude alone
has therefore not been decisively excluded by that control. The negative-control
null is not rejected, which does not prove that its methods are equivalent.

F1, BCE, convergence, runtime, memory and weight-distance bins are descriptive,
without additional superiority tests. No seeds were dropped or replaced, no
hyperparameters selected from tests, and no post-hoc success threshold introduced.
''')
    put(DOC/'RESULTS.md',f'''# Experiment 04 results

Status: **COMPLETED**. {evidence}

Ten independent paired confirmatory seeds completed all 80 fits and 240 predefined
test-endpoint evaluations. The positive primary difference is statistically supported
under the preregistered test assumptions, but the 2 pp practical-support criterion is unmet.

{finaltable}

![Paired test results and validation convergence]({prefix}results/{RUN.name}/experiment04_summary.png)

## Answers to the six research questions

1. **Aligned versus scrambled:** on the specified structured task, 77.539% versus
   76.406%, mean paired +1.133 pp, 95% CI [0.219, 2.046], one-sided p=.01465.
   Seven of ten seed differences are positive; three are negative. The practical threshold is unmet.
2. **Independent generalization:** yes, this measured difference is on 2,048 new examples
   per seed generated only after fitting all models. It establishes within-DGP generalization,
   not transfer to other tasks or distributions.
3. **Alignment versus magnitude:** the primary row-energy-matched comparison is positive.
   Aligned minus uniform is +0.845 pp, but Holm p=.1172. The structured-minus-negative
   effect is +1.157 pp, Holm p=.04883. These patterns suggest task alignment matters,
   while the uniform control prevents a decisive claim that magnitude is fully ruled out.
4. **After release:** the final primary advantage exists after 480 updates using exact ones.
   Immediate release at fixed weights gives +0.537 pp, CI [-0.178,1.252], Holm p=.25.
   Pre-release gives +1.396 pp, Holm p=.08594. Immediate persistence is uncertain;
   final persistence is observed under the primary protocol. No further tests were added.
5. **Overhead:** aligned mean paired optimization ratio is 1.091 on structured data and
   1.051 on negative data, below the preregistered 1.5 limit. This measures a matched dense
   gating path, not an optimized nn.Linear benchmark, and does not establish deployment cost.
6. **Original hypothesis:** limited task-specific statistical support for fixed, known
   relationship alignment. Practical importance and the uniform contrast remain unresolved.
   This does not justify moving directly to learned associations.

## Dynamics and diagnostics

{mdtable(['Task','Condition','Final train accuracy (%)','Final validation accuracy (%)','Train BCE','Validation BCE'],terminal)}

Training fits the small noisy training set very strongly while validation/test BCE is
much larger. This is consistent with overfitting and overconfident errors under the fixed
budget. It is an observation, not a proven cause of any between-condition difference.
No early stopping was retroactively applied. Initial train/validation metrics and complete
curves are retained for every condition; effective initial scaling differs by design.
Final weight-versus-assigned-distance bins are saved in each fit's `weight_distance.json`.
At the final endpoint all gates are one, so effective and raw final weights coincide.
Those bins describe learned weight patterns; they cannot by themselves prove a geometric mechanism.

The negative-control final aligned-minus-scrambled effect is -0.024 pp, CI [-0.808,0.759],
Holm p=.9570. This null result is reported with the same replication count and methods.

## Measured cost

{runtime}

Total confirmatory training wall time was 141.79 s including per-seed audits and I/O.
Optimization timings include gate generation, finite checks, forward/backward and updates;
they exclude evaluation/checkpoint logging. Condition order rotates by seed. One process
serves all conditions; RSS is sampled at logging points and includes shared interpreter,
allocator and dataset state. It is not a per-model peak allocation estimate. Timing is
one local execution without independent hardware benchmarking or repeated timing trials.
''')
    put(DOC/'LIMITATIONS.md',f'''# Limitations and recommendation

- One designed, favorable five-group DGP; no real-world transfer or distribution-shift study.
- Ten independent seeds; limited precision and distributional assumptions for inference.
- Exact sign-flip inference assumes symmetry; it is not valid for every possible mean-zero distribution.
- The primary gain is 1.133 pp. Neither its point estimate nor the lower confidence bound
  reaches the 2 pp practical criterion. The interval still includes effects slightly above 2 pp.
- Uniform-control evidence does not survive Holm adjustment. Scrambling changes topology
  as well as source alignment, so a uniquely geometric causal mechanism is not established.
- Raw initial weights match; effective initial magnitudes and gradient scaling vary with gates.
  Initial metrics are diagnostics, not causal identification of initialization versus dynamics.
- Fixed budget overfits noisy labels; no validation selection or early stopping was used.
  A different shared, preregistered regularization protocol could change the result.
- Hidden-unit coordinates are a prior, not ground-truth learned semantic locations.
- Negative-control permutation can preserve chance local relationships. A null result there
  does not establish equivalence, and datasets are intentionally paired across task variants.
- Immediate-release persistence is uncertain after multiplicity adjustment.
- Runtime baseline retains ones multiplications. Dense gates do not imply sparse acceleration.
  Sampled process RSS does not isolate model memory or a true process peak.
- No Git revision exists; checkout was waived. Source hashes and a local snapshot provide provenance.
- Local preregistration is not an independently timestamped public registration.
- Mermaid syntax and source references were checked; no formal model checking was performed.
  Installed pstack needs unavailable TLC/mise; its bootstrap would install software, which was prohibited.

Recommendation: **do not advance directly to learned associations on this evidence**.
First run a separately preregistered replication of fixed geometry across new interaction
families, with a shared validation-selected budget and an adequately powered uniform
comparison. Do not reuse these test outcomes to claim a new confirmatory success.
Nothing here tests general intelligence, associative memory, consciousness or transformer replacement.
''')
    put(DOC/'TEST_REPORT.md',f'''# Test and completion report

{tests}

New suite command, from repository root:

```powershell
python -m pytest experiments/experiment_04/tests -q -p no:cacheprovider --junitxml=experiments/experiment_04/results/unit_junit.xml
```

Relevant historical suite, from `conn_experiment_01`:

```powershell
python -m pytest tests/test_gates.py tests/test_schedules.py tests/test_models.py tests/test_data.py tests/test_diagnostics.py tests/test_reproducibility.py -q -p no:cacheprovider --junitxml=../experiments/experiment_04/results/regression_junit.xml
```

Reports: [unit XML]({prefix}results/unit_junit.xml), [regression XML]({prefix}results/regression_junit.xml).
This is a targeted regression suite for reused behavior, not the entire historical project suite.

Coverage includes shape/range/zero/monotonicity, all-step row multisets and energy,
uniform energy, weight multiplication, numerical gradients and detach, exact release,
invalid configs, DGP reproducibility and interaction, independent split examples,
noise, label consistency, absence of label inputs to coordinates/permutation,
matching parameter/init/data/order/optimizer state, finite short fitting, known metrics,
unique outputs, bitwise training replay and injected failure/interruption records.

One real smoke attempt failed before fitting on float32 audit accumulation; it is preserved
as `run_20261121T031540_242324Z_45abfa0e`. Float64 diagnostic accumulation fixed the
technical issue without changing any gate, configuration or success threshold. The
fixture was added to the tests. See [technical log]({prefix}TECHNICAL_LOG.md).
Intentional exception fixtures in temporary directories are passing tests, not failed studies.

The subsequent smoke completed eight fits and 24 endpoint metrics. Pilot completed
16 fits and deliberately left its test split unread. Confirmatory completed 80 fits
and 240 endpoint metrics, all finite. Artifact verification recomputes saved prediction
metrics, checks 160 checkpoints, validates pairing and original-file hashes, and writes
[integrity.json]({prefix}results/{RUN.name}/integrity.json).

Eight Mermaid files parsed in pinned Mermaid 11.12.1. All referenced Python symbols
were verified via AST. Seven flowcharts rendered to SVG/PNG; the standard statechart
was parsed only. See [validation](diagrams/validation.json). No formal model checking.
No VS Code Agent notification tool was exposed; the final written response is the completion notification.
''')
    put(DOC/'REPRODUCTION.md',f'''# Reproduction

Use the existing environment, from `E:\\CoNN`. No packages were installed.
Recorded environment: Python 3.10.1, PyTorch 2.5.1+cu121, NumPy 1.26.4,
SciPy 1.15.2, pytest 8.3.5; CPU execution despite available CUDA.
`conn_experiment_01/.venv` has no SciPy, and the separate Python 3.11 launcher has no Torch.
The default `python` executable recorded in every run is the environment used.
Hardware, system, full version provenance and imported source hashes appear in
each run's `environment.json`. Bitwise replay is tested within this environment;
cross-version or cross-machine bitwise identity is not promised.

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH='E:\\CoNN\\experiments\\experiment_04\\src'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
python -m pytest experiments/experiment_04/tests -q -p no:cacheprovider
python -m conn_exp04.runner train --config experiments/experiment_04/config/smoke.json
# Use the unique RUN path printed above:
python -m conn_exp04.runner evaluate --run <SMOKE_RUN>
python -m conn_exp04.runner train --config experiments/experiment_04/config/pilot.json
# Pilot is for training/validation checks; do not evaluate its test split.
python -m conn_exp04.runner train --config experiments/experiment_04/config/confirmatory.json
python -m conn_exp04.runner evaluate --run <CONFIRMATORY_RUN>
python -m conn_exp04.analysis --run <CONFIRMATORY_RUN>
python experiments/experiment_04/scripts/verify_artifacts.py <CONFIRMATORY_RUN>
```

Config loader validates all supplied fields; no hidden tuning phase or seed replacement exists.
Each train command creates a timestamp-plus-UUID directory. Do not rerun `prepare.py`;
it intentionally refuses to overwrite initial configurations. Evaluation refuses an
already evaluated run. Analysis refuses an existing analysis file. Reproduction uses
a new run directory, preserving original measurements. Failed runs remain failed;
there is no silent checkpoint resume.

Frozen main config SHA-256:
`ff17dd4e5ba36c4eec8e63f11b9db4489611fa0ee1317203d280f5613d67ff42`.
The immutable run config, original source snapshot, initial states, batches, gate
permutations/distances, train/validation samples, checkpoints, optimizer state,
test labels and logits allow audit without retraining. Seeds for data/model/order/
mask/training are recorded separately in each fit's provenance. Test RNG seeds
and sample hashes appear in endpoint rows. Saved checkpoints belong to this
locally generated run; no untrusted historical checkpoint was loaded.

To validate diagrams again, `python experiments/experiment_04/scripts/validate_diagrams.py`.
It uses existing Playwright and local Chrome plus a pinned Mermaid browser script
from jsDelivr. It installs no packages, uploads no diagram content, and blocks network
requests after the renderer loads. The statechart is parsed without a pstack render.
The diagram generator and documentation report intentionally create files exclusively;
rebuild them into a new destination or preserve outputs before a deliberate regeneration.
''')
    catalog=json.loads((DOC/'diagrams'/'catalog.json').read_text())
    put(DOC/'README.md',f'''# Experiment 04 — Relationship Geometry Validation

**COMPLETED**: implementation, mathematical tests, paired smoke, two-seed pilot,
ten-seed confirmatory experiment and held-out evaluation.

Aligned final accuracy: **77.54%**, scrambled: **76.41%**; paired effect **+1.13 pp**,
95% CI **[0.22, 2.05] pp**, preregistered one-sided p=**.01465**.
The 2 pp practical-support criterion is unmet. The uniform contrast does not survive
Holm adjustment. Recommendation: strengthen fixed-geometry evidence before learned associations.

- [Design](EXPERIMENT_DESIGN.md)
- [Mathematics](MATHEMATICAL_SPECIFICATION.md)
- [Actual results and six research answers](RESULTS.md)
- [Paired statistics](STATISTICAL_ANALYSIS.md)
- [Tests and integrity](TEST_REPORT.md)
- [Limitations and recommendation](LIMITATIONS.md)
- [Reproduction](REPRODUCTION.md)
- [Preregistration]({prefix}PREREGISTRATION.md)

## Diagrams and verified sources

'''+ '\n'.join(f"- [{r['title']}](diagrams/{r['name']}.mmd)" for r in catalog)+f'''

Source: [core.py]({prefix}src/conn_exp04/core.py), [runner.py]({prefix}src/conn_exp04/runner.py),
[analysis.py]({prefix}src/conn_exp04/analysis.py), and linked unmodified historical modules.
All diagrams were parsed; Python names were verified. No formal model checking.

{evidence}
''')
    put(BASE/'README.md',f'''# Experiment 04 — Relationship Geometry Validation

Isolated implementation of a preregistered four-condition comparison on a known
latent-interaction task and a negative geometry control. No original CoNN source changed.

**Completed:** 29 new tests + 32 historical regressions passed; smoke 1 seed / 8 fits;
pilot 2 seeds / 16 fits (test unread); confirmatory 10 seeds / 80 fits / 240 test endpoints.

Aligned versus scrambled final test accuracy: 77.54% versus 76.41%; paired +1.13 pp,
95% CI [0.22,2.05] pp, one-sided exact p=.01465. Practical 2 pp criterion unmet.
Uniform secondary contrast: Holm p=.1172. No learned associations were implemented.

[Full report](../../docs/research_validation/experiment_04/README.md),
[preregistration](PREREGISTRATION.md), [technical log](TECHNICAL_LOG.md).

Structure: `config/` frozen JSON; `src/conn_exp04/` DGP/controls/runner/statistics;
`tests/` automated checks; `scripts/` preparation, figures/report, diagrams and verification;
`results/` unique run directories with frozen config, provenance, checkpoints and measurements.
Use the [reproduction instructions](../../docs/research_validation/experiment_04/REPRODUCTION.md).
The user waived Git checkout; all original source/document/config hashes remain unchanged.
''')
    put(BASE/'results'/'stage_manifest.json',json.dumps({'stages':RUNS,'failed_smoke':'run_20261121T031540_242324Z_45abfa0e',
        'unit_tests':unit,'regression_tests':reg,'pilot_test_accessed':False,'confirmatory_fits':80,
        'confirmatory_endpoint_metrics':240},indent=2))
    print('Created eight research documents plus experiment README, figure, CSV exports and stage manifest.')


if __name__=='__main__': main()
