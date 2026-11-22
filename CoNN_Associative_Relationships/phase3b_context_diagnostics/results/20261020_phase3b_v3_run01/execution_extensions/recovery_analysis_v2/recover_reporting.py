"""Versioned, analysis-only recovery of a lossless NumPy-bool JSON defect.

Frozen original training/report source and all failed-stage evidence stay intact.
No training, optimizer calls, endpoint changes or new bootstrap choices.
"""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from pathlib import Path
import sys
sys.stdout.reconfigure(encoding="utf-8",errors="strict")
sys.stderr.reconfigure(encoding="utf-8",errors="strict")
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import importlib.util
import json
import shutil
import numpy as np
from common import read,write,sha

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.ROOT=ROOT
    return module

def main():
    run=ROOT/'results/20261020_phase3b_v3_run01';repair=run/'execution_extensions/recovery_analysis_v2'
    if (repair/'completion.json').exists():print(read(repair/'completion.json'));return
    repair.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,repair/'recover_reporting.py')
    assert read(run/'verification.json')['verified_fits']==465
    assert read(run/'execution_status.json')['stage']=='completed'
    failures=list((ROOT/'results/failures').glob('*.json'))
    assert len(failures)==1 and read(failures[0])['exception']=="TypeError('Object of type bool is not JSON serializable')"
    assert 'reporting.py' in read(failures[0])['traceback']
    for rel,digest in read(run/'source_manifest.json').items():assert sha(ROOT/rel)==digest,rel
    evidence=repair/'evidence';evidence.mkdir()
    for folder in ('analysis','plots'):
        if (run/folder).exists():shutil.copytree(run/folder,evidence/folder)
    shutil.copytree(ROOT/'reports',evidence/'reports')
    for name in ('analysis_summary.json','research_report.md','report_manifest.json','execution_status.json','verification.json'):
        if (run/name).exists():shutil.copy2(run/name,evidence/name)
    failurepaths=failures+list((run/'execution_extensions').glob('*/failure.json'))
    write(repair/'acknowledged_failure_evidence.json',{str(p):{'sha256':sha(p),'record':read(p)} for p in failurepaths})
    codes=repair/'code';codes.mkdir()
    original=ROOT/'src/reporting.py';text=original.read_text(encoding='utf-8')
    token='descriptive_ci_excludes_zero_positive=lo>0'
    assert text.count(token)==1
    fixed=codes/'reporting_json_native.py';fixed.write_text(text.replace(token,'descriptive_ci_excludes_zero_positive=bool(lo>0)'),encoding='utf-8')
    copies={}
    for name in ('measured_summary.py','forward_rank_sensitivity.py','final_qa.py'):
        source=(ROOT/'scripts'/name).read_text(encoding='utf-8')
        source=source.replace('ROOT=Path(__file__).resolve().parents[1]',f'ROOT=Path({str(ROOT)!r})')
        if name=='measured_summary.py':
            source=source.replace('finalization_driver_v1/completion.json','recovery_analysis_v2/finalization_stage_complete.json')
            source=source.replace("execution_extensions/measured_summary_v1","execution_extensions/measured_summary_recovery_v1")
            source=source.replace('No post-seal scientific repair or outcome-driven redesign occurred.',
                'An analysis-only JSON serialization defect was repaired losslessly by converting one NumPy boolean to a Python boolean. Training, endpoint calculations, bootstrap draws and CSV statistics were unchanged. Downstream report guards stopped; all failure evidence was retained. No scientific redesign occurred.')
        elif name=='forward_rank_sensitivity.py':
            source=source.replace('execution_extensions/forward_rank_sensitivity_v1','execution_extensions/forward_rank_sensitivity_recovery_v1')
            source=source.replace('execution_extensions/measured_summary_driver_v1/completion.json','execution_extensions/recovery_analysis_v2/measured_stage_complete.json')
        else:
            source=source.replace('execution_extensions/final_qa_v1','execution_extensions/final_qa_recovery_v1')
            source=source.replace('execution_extensions/forward_rank_sensitivity_v1/completion.json','execution_extensions/forward_rank_sensitivity_recovery_v1/completion.json')
        path=codes/name;path.write_text(source,encoding='utf-8');copies[name]=path
    checks=[json.loads(json.dumps(bool(np.bool_(True)))) is True,
        json.loads(json.dumps(bool(np.bool_(False)))) is False,
        float(np.float64(.00011565544771546819))==.00011565544771546819,
        'bool(lo>0)' in fixed.read_text(),sha(original)==read(run/'source_manifest.json')['src\\reporting.py']]
    checks.append(sys.stdout.encoding.lower().replace("-","")=="utf8")
    assert all(checks)
    write(repair/'repair_tests.json',{'passed':len(checks),'failed':0,'optimizer_updates':0,
        'scope':'boolean true/false roundtrip, unchanged float, exact patch, preserved frozen source'})
    write(repair/'manifest.frozen.json',{'source_sha256':sha(__file__),'training_source_manifest_sha256':sha(run/'source_manifest.json'),
        'reporting_original_sha256':sha(original),'reporting_repair_sha256':sha(fixed),
        'repair':'one lossless Python bool conversion at JSON serialization boundary',
        'postprocessing_changes':'new stage directories and acknowledged recovery prerequisites only; numerical algorithms unchanged',
        'copied_source_sha256':{n:sha(p) for n,p in copies.items()},'optimizer_updates':0})
    log=ROOT/'reports/REPAIR_AND_DEVIATION_LOG.md'
    with log.open('a',encoding='utf-8') as f:
        f.write('\nPost-training reporting defect: full training and independent verification succeeded, but final analysis_summary JSON could not serialize a NumPy boolean from lo>0. All CSV statistics and reports had already been generated. Report guards stopped dependent stages. Failure JSONs, generated analysis, plots and reports were preserved under execution_extensions/recovery_analysis_v2/evidence; original failure files remain in place. A separately snapshotted reporting copy converts exactly that boolean to Python bool. Six repair checks passed, including explicit UTF-8 console configuration. The earlier recovery stopped only on a legacy cp1252 console print; UTF-8 artifacts and all numeric CSVs were intact and unchanged. No frozen original source, model, data, endpoint, bootstrap stream, fit or optimizer update changed. Recovery uses new stage identifiers and explicit acknowledged prerequisites. Before/after contrast/subject/seed CSV hashes must be identical.\n')
    keys=['paired_contrasts.csv','subject_differences.csv','seed_differences.csv']
    before={n:sha(run/'analysis'/n) for n in keys}
    load(fixed,'reporting_json_native').report(run)
    after={n:sha(run/'analysis'/n) for n in keys}
    assert before==after,'Statistics changed during serialization-only repair'
    write(repair/'statistic_identity.json',{'before':before,'after':after,'bitwise_csv_identity':True})
    assert read(run/'analysis_summary.json')['complete']
    load(ROOT/'scripts/finalize_analysis.py','finalize_original').main()
    write(repair/'finalization_stage_complete.json',{'completed_via':'recovery_analysis_v2','original_failed_driver_preserved':True})
    load(copies['measured_summary.py'],'measured_recovery').main()
    write(repair/'measured_stage_complete.json',{'completed_via':'recovery_analysis_v2'})
    load(copies['forward_rank_sensitivity.py'],'forward_rank_recovery').main()
    load(copies['final_qa.py'],'final_qa_recovery').main()
    write(repair/'completion.json',{'analysis_recovery_completed':True,'repair_checks_passed':6,
        'training_unchanged':True,'csv_statistics_unchanged':True,'optimizer_updates':0})
    write(run/'delivery_manifest.json',{str(p.relative_to(run)):sha(p) for p in run.rglob('*') if p.is_file() and p.name!='delivery_manifest.json'})
    print('Analysis-only recovery completed; all original failures preserved',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        import traceback
        run=ROOT/'results/20261020_phase3b_v3_run01'
        write(run/'execution_extensions/recovery_analysis_v2/failure.json',{'exception':repr(exc),'traceback':traceback.format_exc()})
        raise
