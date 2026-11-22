"""Final prose/figure readability review; never changes measurements."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from pathlib import Path
import sys
sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import re
import shutil
import time
import numpy as np
from common import read,write,sha

def prose(text):
    exempt={'float','int','uint','sigma','C','D','G','H','TF','SHA','e','E'}
    text=re.sub(r'([A-Za-z]+)(\d)',lambda m:m[1]+('' if m[1] in exempt else ' ')+m[2],text)
    text=re.sub(r'(?<=[A-Za-z)]),\s*(?=[A-Za-z0-9])',', ',text)
    text=text.replace('bootstrapCI','bootstrap CI').replace('The.001','The .001')
    text=re.sub(r',(?=[+−-])',', ',text)
    return text

def polish(text):
    pieces=re.split(r'(`[^`]*`)',text)
    return ''.join(p if p.startswith('`') else prose(p) for p in pieces)

def main():
    run=ROOT/'results/20261020_phase3b_v3_run01';ext=run/'execution_extensions/delivery_review_v2'
    ext.mkdir(parents=True,exist_ok=False);shutil.copy2(__file__,ext/'delivery_review.py')
    write(ext/'manifest.frozen.json',{'source_sha256':sha(__file__),'optimizer_updates':0,
        'scope':'prose whitespace only, precise figure colorbar labels and explicit repair metadata; no changed data/statistics'})
    deadline=time.monotonic()+21600
    while not (run/'execution_extensions/recovery_analysis_v2/completion.json').exists():
        error=run/'execution_extensions/recovery_analysis_v2/failure.json'
        if error.exists():raise RuntimeError(f'Unresolved reporting recovery failure: {read(error)}')
        if time.monotonic()>deadline:raise TimeoutError('Delivery review wait exceeded six hours')
        time.sleep(10)
    completion=read(run/'completion.json');primary=completion['primary']
    evidence=ext/'before';evidence.mkdir()
    reportpaths=[run/'research_report.md',ROOT/'reports/PHASE3B_FINAL_REPORT.md',run/'reports/PHASE3B_FINAL_REPORT.md',
        ROOT/'reports/PHASE3B_EXECUTIVE_SUMMARY.md',run/'reports/PHASE3B_EXECUTIVE_SUMMARY.md']
    hashes={}
    for i,path in enumerate(reportpaths):
        old=path.read_text(encoding='utf-8');new=polish(old)
        shutil.copy2(path,evidence/f'{i}_{path.name}')
        # Every inline code/hash block and decimal token remains exactly intact.
        assert re.findall(r'`[^`]*`',old)==re.findall(r'`[^`]*`',new)
        assert re.findall(r'[+−-]?\d+\.\d+(?:[eE][−+-]?\d+)?',old)==re.findall(r'[+−-]?\d+\.\d+(?:[eE][−+-]?\d+)?',new)
        hashes[str(path)]={'before':sha(path)};path.write_text(new,encoding='utf-8');hashes[str(path)]['after']=sha(path)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,3,figsize=(13,8),layout='constrained')
    for col,cell in enumerate(('C0','C2','C4')):
        with np.load(run/'pilots'/f'gaussian_{cell}_seed400_fold0'/'attenuation.npz') as p:
            a=p['ih_initial_0'][:,:128];b=p['ih_final_0'][:,:128]
        lo=min(a.min(),b.min());hi=max(a.max(),b.max())
        for row,data,label in [(0,a,'initial A'),(1,b,'final A')]:
            im=axes[row,col].imshow(data,aspect='auto',vmin=lo,vmax=hi)
            axes[row,col].set_title(cell+' '+label)
            fig.colorbar(im,ax=axes[row,col],format=('%.9f' if cell=='C0' else '%.8f' if cell=='C2' else '%.6f'))
        delta=b-a;lim=max(float(np.abs(delta).max()),1e-20)
        im=axes[2,col].imshow(delta,aspect='auto',vmin=-lim,vmax=lim,cmap='coolwarm')
        axes[2,col].set_title(cell+' ΔA; separate delta scale');fig.colorbar(im,ax=axes[2,col],format='%.1e')
    fig.suptitle('Reference attenuation, Gaussian pilot: initial/final share scales within each cell',fontsize=13)
    fig.savefig(run/'plots/historical_revised_attenuation_readable.png',dpi=150);plt.close(fig)
    note='\n\nThe final visual review retained earlier renders and added plots/historical_revised_attenuation_readable.png with sufficient colorbar precision to distinguish tiny attenuation changes. Prose spacing was improved without altering any decimal value, inline code/hash, measurement or statistical decision. Reporting-only JSON/console repairs and their stopped-stage evidence remain fully preserved under recovery_analysis_v1 and recovery_analysis_v2; six final repair checks passed with zero optimizer updates. See MEASURED_CONTRASTS_AND_IDENTIFICATION_LIMITS.md for the distinction between the modest positive primary estimate and the unmet practical threshold, and for the uncertain exploratory C4−C0 comparison.\n'
    for path in reportpaths[:3]:
        with path.open('a',encoding='utf-8') as f:f.write(note)
    # Synchronize final documentation, including the actual recovery command.
    shutil.copy2(ROOT/'README.md',run/'README.md')
    for path in (ROOT/'reports').glob('*.md'):shutil.copy2(path,run/'reports'/path.name)
    completion['reporting_repair']={'resolved':True,'types':['NumPy bool JSON','legacy Windows console encoding'],
        'recovery':'execution_extensions/recovery_analysis_v2','repair_checks_passed':6,
        'contrast_subject_seed_csvs_byte_identical':True,'optimizer_updates':0,'all_failure_evidence_preserved':True}
    completion['final_report_sha256']=sha(run/'reports/PHASE3B_FINAL_REPORT.md')
    completion['executive_summary_sha256']=sha(run/'reports/PHASE3B_EXECUTIVE_SUMMARY.md')
    assert f"{primary['mean']:+.8f}" in (run/'research_report.md').read_text(encoding='utf-8')
    write(run/'completion.json',completion)
    write(ext/'completion.json',{'prose_and_figure_review_passed':True,'decimal_tokens_and_inline_code_unchanged':True,
        'report_hashes':hashes,'optimizer_updates':0})
    write(run/'delivery_manifest.json',{str(p.relative_to(run)):sha(p) for p in run.rglob('*') if p.is_file() and p.name!='delivery_manifest.json'})
    print(completion,flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        import traceback
        run=ROOT/'results/20261020_phase3b_v3_run01'
        write(run/'execution_extensions/delivery_review_v2/failure.json',{'exception':repr(exc),'traceback':traceback.format_exc()})
        raise
