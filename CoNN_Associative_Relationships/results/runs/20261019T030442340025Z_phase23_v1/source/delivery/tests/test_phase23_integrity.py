"""Critical guardrails: no output outside experiment, no tampered frozen config."""
import json
from pathlib import Path
import shutil
import uuid

import pytest

from conn_assoc.paths import EXPERIMENT_ROOT
from conn_assoc.phase23_core import train_job


def test_training_output_rejects_other_workspace_or_temporary_directory(tmp_path):
    with pytest.raises(ValueError,match='inside this experiment'):
        train_job(tmp_path,'P2_dense',400,0,.01,steps=1)
    assert not list(tmp_path.iterdir())


def test_training_rejects_modified_frozen_configuration():
    active=Path((EXPERIMENT_ROOT/'config/phase23_active_run.txt').read_text().strip())
    run=active/'verification'/('tamper_guard_'+uuid.uuid4().hex);run.mkdir(parents=True)
    for name in ('config.frozen.json','protocol_seal.json'):shutil.copy2(active/name,run/name)
    cfg=json.loads((run/'config.frozen.json').read_text());cfg['steps']=1
    (run/'config.frozen.json').write_text(json.dumps(cfg))
    with pytest.raises(RuntimeError,match='Config seal changed'):
        train_job(run,'P2_dense',400,0,.01,steps=1)
    assert not (run/'jobs').exists()
