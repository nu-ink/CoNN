"""Seal a fully completed delivery; never used to mark partial work complete."""
from datetime import datetime,timezone
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:]=[p for p in sys.path if Path(p or '.').resolve()!=ROOT/'scripts']
sys.path[:0]=[str(ROOT/'src'),str(ROOT.parent/'conn_experiment_01/src')]
from PIL import Image
from conn_assoc.phase23 import snapshot
from conn_assoc.phase23_core import json_write,sha


def main():
    run=Path((ROOT/'config/phase23_active_run.txt').read_text().strip())
    verification=json.loads((run/'final_verification.json').read_text())
    assert verification['fits']==1050 and verification['optimizer_updates']==3360000
    assert verification['official_test_model_evaluations']==0
    for report in (ROOT/'reports/PHASE_2_3_IMPLEMENTATION_AND_RESULTS.md',run/'research_report.md'):
        for link in re.findall(r'!\[[^\]]*\]\(([^)]+)\)',report.read_text(encoding='utf-8')):
            assert (report.parent/link).resolve().is_file(),(report,link)
    pngs=list((run/'plots').glob('*.png'))
    for p in pngs:
        with Image.open(p) as im:im.verify()
    snapshot(run,'delivery')
    manifest={}
    before=set(json.loads((run/'prior_associative_manifest.json').read_text()))
    for p in ROOT.rglob('*'):
        if not p.is_file():continue
        relative=str(p.relative_to(ROOT))
        if p.suffix in ('.py','.ps1','.md','.yaml','.toml') and not any(s.startswith('.venv') or s.startswith('.pip-cache') for s in p.relative_to(ROOT).parts) and not p.is_relative_to(ROOT/'results'):
            manifest[relative]=sha(p)
    json_write(run/'delivery_source_manifest.json',manifest)
    executions=[]
    for folder in sorted((run/'execution').iterdir()):
        plan=json.loads((folder/'plan.json').read_text())
        progress=json.loads((folder/'progress.json').read_text()) if (folder/'progress.json').exists() else {}
        executions.append({'directory':folder.name,'stage':plan['stage'],'workers':plan['workers'],
            'already_completed':plan['already_completed'],'elapsed_seconds':progress.get('elapsed_seconds'),
            'failed_attempts':progress.get('failed',0),'completion_manifest':(folder/'completion.json').exists()})
    json_write(run/'completion.json',{'status':'development_completed','completed_utc':datetime.now(timezone.utc).isoformat(),
        'fits':1050,'completed_logical_optimizer_updates':3360000,'conditions':10,'folds':7,'engineering_seeds':5,
        'learning_rates_per_condition':3,'official_test_model_evaluations':0,'phase4_implemented':False,
        'execution_attempts':executions,'pngs_verified':len(pngs),'report_sha256':sha(ROOT/'reports/PHASE_2_3_IMPLEMENTATION_AND_RESULTS.md'),
        'successful_grid_io_retry_events':sum(len(p.read_text().splitlines()) for p in (run/'jobs').glob('*/io_retries.jsonl')),
        'final_verification_sha256':sha(run/'final_verification.json'),'limitations':'Engineering CV with fixed-scaler preprocessing overlap and selection reuse; not independent confirmation.'})
    inventory=run/'new_file_inventory.json'
    new=[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size} for p in ROOT.rglob('*')
         if p.is_file() and str(p.relative_to(ROOT)) not in before and '__pycache__' not in p.parts and p!=inventory]
    new.append({'path':str(inventory.relative_to(ROOT)),'bytes':None,'note':'Self-referential inventory size deliberately omitted'})
    json_write(inventory,new)
    print('Sealed completed development run:',run)


if __name__=='__main__':main()
