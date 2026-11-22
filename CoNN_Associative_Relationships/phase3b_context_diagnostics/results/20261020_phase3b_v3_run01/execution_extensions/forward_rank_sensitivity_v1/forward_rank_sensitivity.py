"""Existing-checkpoint precision sensitivity; zero optimizer updates."""
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
from common import read,write,sha,csv_write,csv_read
from models import Model,configure,factor_reference

def main():
    run=ROOT/'results/20261020_phase3b_v3_run01'
    extension=run/'execution_extensions/forward_rank_sensitivity_v1';extension.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,extension/'forward_rank_sensitivity.py')
    write(extension/'manifest.frozen.json',{'source_sha256':sha(__file__),'scope':'existing checkpoint diagnostic sensitivity only',
        'optimizer_updates':0,'frozen_primary_analysis_unchanged':True,'tolerance_relative':1e-10,
        'definitions':['float64 factor reference attenuation','actual float32 forward C then float64 A/SVD']})
    deadline=time.monotonic()+21600
    while not (run/'execution_extensions/measured_summary_driver_v1/completion.json').exists():
        failures=list((run/'execution_extensions').glob('*/failure.json'))
        if failures:raise RuntimeError(f'Preserved earlier failure: {failures}')
        if time.monotonic()>deadline:raise TimeoutError('Forward-rank reporting wait exceeded six hours')
        time.sleep(10)
    configure('cpu');rows=[]
    for cp in sorted(run.glob('*/**/completion.json')):
        job=cp.parent
        if not (job/'predictions.npz').exists():continue
        done=read(cp)
        if done['condition'].endswith('masked'):continue
        model=Model(done['condition'],done['seed'])
        for phase in ('initial','final'):
            state=torch.load(job/(phase+'.pt'),map_location='cpu',weights_only=False)['model'];model.load_state_dict(state)
            with torch.no_grad():
                for layer in ('ih','hh'):
                    factor=getattr(model,'context_'+layer)
                    for context in range(3):
                        forward=1-factor(context).double()
                        reference=1-factor_reference(factor,context)
                        sv=torch.linalg.svdvals(forward);ref=torch.linalg.svdvals(reference)
                        rows.append({'stage':job.parent.name,'condition':done['condition'],'seed':done['seed'],'fold':done['fold'],
                            'phase':phase,'layer':layer,'context':context,'forward_sigma1':float(sv[0]),'forward_sigma2':float(sv[1]),
                            'forward_sigma3':float(sv[2]),'forward_sigma2_over_sigma1':float(sv[1]/sv[0]),
                            'forward_numerical_rank':int((sv>sv[0]*1e-10).sum()),
                            'reference_sigma2_over_sigma1':float(ref[1]/ref[0]),'reference_numerical_rank':int((ref>ref[0]*1e-10).sum()),
                            'max_abs_forward_reference_difference':float((forward-reference).abs().max())})
    csv_write(run/'analysis/forward_vs_reference_rank.csv',rows)
    text='# Forward versus reference attenuation rank\n\n'
    text+='No model was trained for this sensitivity analysis. It reads saved initial/final checkpoints only. The original rank and matrix-difference diagnostics recompute sigmoid factors and C in float64, isolating the mathematical factorization. This supplement also evaluates C in its actual forward dtype, then converts that C to float64 before constructing A=1−C and computing SVD. Both use sigma1×1e−10 numerical-rank tolerance.\n\n'
    text+='Float32 elementwise rounding can introduce tiny extra singular values into the forward matrix. Its numerical rank at this strict tolerance is therefore not synonymous with independent rank-factor capacity. H1 concerns the unquantized parameterization: identical columns restrict its attenuation to rank at most one; distinct columns can supply a second component. Neither a rounding-induced higher numerical rank nor a mathematically nonzero second component establishes learned utility. The frozen primary statistics and Level2 algorithm are unchanged.\n\n'
    text+='| Gaussian cell | Phase | Mean reference sigma2/sigma1 | Mean actual-forward sigma2/sigma1 | Reference rank range | Forward rank range |\n| --- | --- | --- | --- | --- | --- |\n'
    for c in ('C0','C1','C2','C4'):
        for phase in ('initial','final'):
            rr=[r for r in rows if r['stage']=='grid' and r['condition']=='gaussian_'+c and r['phase']==phase]
            text+=f"| {c} | {phase} | {np.mean([r['reference_sigma2_over_sigma1'] for r in rr]):.8g} | {np.mean([r['forward_sigma2_over_sigma1'] for r in rr]):.8g} | {min(r['reference_numerical_rank'] for r in rr)}–{max(r['reference_numerical_rank'] for r in rr)} | {min(r['forward_numerical_rank'] for r in rr)}–{max(r['forward_numerical_rank'] for r in rr)} |\n"
    text+='\nAll completed contextual conditions and both retained C3 pilots are included in the CSV, including separate layers/contexts. Forward-level learning claims still use actual float32 live-support changes and output interventions, rather than reference rank alone.\n'
    for path in (ROOT/'reports/FORWARD_RANK_SENSITIVITY.md',run/'reports/FORWARD_RANK_SENSITIVITY.md'):path.write_text(text,encoding='utf-8')
    addition='\n\n## Forward-dtype rank sensitivity\n\nThe reference rank in section5 is computed from factors in float64. Actual forward C is float32; converting it to float64 does not remove its prior rounding. The additional saved-checkpoint diagnostic reports both ranks at the same tolerance. Tiny rounding-induced singular values are not additional independent learned factors. See FORWARD_RANK_SENSITIVITY.md and analysis/forward_vs_reference_rank.csv. The frozen endpoint, uncertainty analysis and Level2 computation are unchanged.\n'
    for path in (run/'research_report.md',ROOT/'reports/PHASE3B_FINAL_REPORT.md',run/'reports/PHASE3B_FINAL_REPORT.md'):
        with path.open('a',encoding='utf-8') as f:f.write(addition)
    write(extension/'completion.json',{'checkpoint_matrices_measured':len(rows),'optimizer_updates':0,'primary_analysis_unchanged':True})
    write(run/'delivery_manifest.json',{str(p.relative_to(run)):sha(p) for p in run.rglob('*') if p.is_file() and p.name!='delivery_manifest.json'})
    print('Forward/reference rank matrices measured:',len(rows),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        import traceback
        run=ROOT/'results/20261020_phase3b_v3_run01'
        write(run/'execution_extensions/forward_rank_sensitivity_v1/failure.json',{'exception':repr(exc),'traceback':traceback.format_exc()})
        raise
