"""Final artifact checks and measured loss rendering, no optimization."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import shutil
import time
import numpy as np
import torch
from common import read,write,sha,csv_read
from models import configure

def main():
    run=ROOT/'results/20261020_phase3b_v3_run01'
    ext=run/'execution_extensions/final_qa_v1';ext.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,ext/'final_qa.py')
    write(ext/'manifest.frozen.json',{'source_sha256':sha(__file__),'optimizer_updates':0,
        'role':'integrity checks and existing-measurement loss rendering','primary_analysis_unchanged':True})
    deadline=time.monotonic()+21600
    while not (run/'execution_extensions/forward_rank_sensitivity_v1/completion.json').exists():
        failure=list((run/'execution_extensions').glob('*/failure.json'))
        if failure:raise RuntimeError(f'Preserved prior failure: {failure}')
        if time.monotonic()>deadline:raise TimeoutError('Final QA wait exceeded six hours')
        time.sleep(10)
    configure('cpu')
    for rel,digest in read(run/'source_manifest.json').items():assert sha(ROOT/rel)==digest,rel
    seal=read(run/'protocol_seal.json');assert sha(run/'config.frozen.json')==seal['config_sha256']
    assert sha(run/'source_manifest.json')==seal['source_manifest_sha256']
    verified=read(run/'verification.json');assert verified['verified_fits']==465 and verified['verified_logical_updates']==1488000
    finite=0
    for cp in sorted(run.glob('*/**/completion.json')):
        job=cp.parent
        if not (job/'predictions.npz').exists():continue
        st=torch.load(job/'final.pt',map_location='cpu',weights_only=False)
        assert st['next_step']==3200 and st['executed']==list(range(3200))
        assert all(bool(torch.isfinite(v).all()) for v in st['model'].values()),job
        finite+=1
    assert finite==465
    assert read(run/'final_preservation_audit/historical_integrity.json')['passed']
    report=(ROOT/'reports/PHASE3B_FINAL_REPORT.md').read_text(encoding='utf-8')
    assert seal['protocol_sha256'] in report and seal['config_sha256'] in report
    primary=read(run/'analysis_summary.json')['primary']
    assert f"{primary['mean']:+.8f}" in report and f"{primary['ci_low']:+.8f}" in report and f"{primary['ci_high']:+.8f}" in report
    release=csv_read(run/'analysis/release_vs_optimization.csv')
    identity=csv_read(run/'analysis/release_checkpoint_identity.csv')
    assert len(identity)==455 and all(r['weights_bitwise_identical']=='True' for r in identity)
    conditions=list(dict.fromkeys(r['condition'] for r in release))
    text='# Loss and dense-release diagnostics\n\n'
    text+='All values below are measured checkpoint evaluations, averaged equally across35fits and three contexts. These pooled-window losses/accuracies differ from the primary equal-subject macro-F1 endpoint. Pre/post evaluations use bitwise-identical weights at optimizer step2240; final values additionally include960updates. No missing step or fit is interpolated.\n\n'
    text+='| Condition | Pre loss | Post loss | Final loss | Gate-switch loss change | Subsequent optimization loss change |\n| --- | --- | --- | --- | --- | --- |\n'
    for c in conditions:
        rows=[r for r in release if r['condition']==c and r['split']=='val']
        pre=np.mean([float(r['pre_loss']) for r in rows]);post=np.mean([float(r['post_loss']) for r in rows]);final=np.mean([float(r['final_loss']) for r in rows])
        text+=f'| {c} | {pre:.8f} | {post:.8f} | {final:.8f} | {post-pre:+.8f} | {final-post:+.8f} |\n'
    text+='\nThe loss chart uses observed curve checkpoints only, averaged over completed fits/contexts. It does not replace the subject endpoint or establish that widening is generally helpful or harmful.\n'
    for path in (ROOT/'reports/LOSS_AND_RELEASE_DIAGNOSTICS.md',run/'reports/LOSS_AND_RELEASE_DIAGNOSTICS.md'):path.write_text(text,encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    curves=csv_read(run/'analysis/learning_curves.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,split in zip(axes,('train','val')):
        for c in ('gaussian_masked','gaussian_C0','gaussian_C2','gaussian_C4'):
            rr=[r for r in curves if r['stage']=='grid' and r['condition']==c and r['split']==split and r['kind'] in ('initial','curve','final')]
            steps=sorted(set(int(r['completed_steps']) for r in rr))
            ax.plot(steps,[np.mean([float(r['loss']) for r in rr if int(r['completed_steps'])==s]) for s in steps],label=c)
        ax.axvline(2240,color='grey',ls=':',label='Dense release');ax.set_title(split+' cross-entropy');ax.set_xlabel('Optimizer updates');ax.set_ylabel('Mean fit/context window loss');ax.legend(fontsize=7)
    fig.savefig(run/'plots/train_validation_loss.png',dpi=150);plt.close(fig)
    addition='\n\n## Final integrity and loss check\n\nAll465completed pilot/grid final model states were checked finite, and each checkpoint contains3200actual recorded optimizer steps. Frozen training-source/config hashes remained unchanged. All455grid pre/post-release weight pairs were bitwise identical. LOSS_AND_RELEASE_DIAGNOSTICS.md and plots/train_validation_loss.png separate the measured gate-switch loss change from subsequent optimization. The final preservation audit passed.\n'
    for path in (run/'research_report.md',ROOT/'reports/PHASE3B_FINAL_REPORT.md',run/'reports/PHASE3B_FINAL_REPORT.md'):
        with path.open('a',encoding='utf-8') as f:f.write(addition)
    status=read(run/'execution_status.json');account=read(run/'optimizer_accounting.json');summary=read(run/'analysis_summary.json')
    write(ext/'completion.json',{'finite_final_model_states':finite,'release_weight_pairs_verified':len(identity),
        'frozen_source_hashes_verified':True,'primary_report_numbers_verified':True,'optimizer_updates':0})
    write(run/'completion.json',{'version':'phase3b_v3','status':'completed','grid_fits':455,'pilot_fits':10,'omitted_grid_fits':70,
        'training_optimizer_updates':1488000,'additional_diagnostic_optimizer_calls':940,'backend':'cpu',
        'unit_tests_passed_distinct':19,'unit_test_runs':2,'unit_tests_failed':0,'official_test_model_evaluations':0,
        'finite_final_model_states':finite,'execution_seconds':status['elapsed_seconds'],'primary':primary,
        'gaussian_C4_level1_fits':summary['gaussian_C4_level1_fits'],'level2_pair_passes':summary['level2_passed_comparisons'],
        'level2_pair_denominator':summary['level2_denominator'],'level3_predictive_threshold_met':summary['level3_predictive_criterion'],
        'historical_preservation_passed':True,'protocol_seal':seal,
        'final_report_sha256':sha(run/'reports/PHASE3B_FINAL_REPORT.md'),'executive_summary_sha256':sha(run/'reports/PHASE3B_EXECUTIVE_SUMMARY.md')})
    write(run/'delivery_manifest.json',{str(p.relative_to(run)):sha(p) for p in run.rglob('*') if p.is_file() and p.name!='delivery_manifest.json'})
    print(read(run/'completion.json'),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        import traceback
        run=ROOT/'results/20261020_phase3b_v3_run01'
        write(run/'execution_extensions/final_qa_v1/failure.json',{'exception':repr(exc),'traceback':traceback.format_exc()})
        raise
