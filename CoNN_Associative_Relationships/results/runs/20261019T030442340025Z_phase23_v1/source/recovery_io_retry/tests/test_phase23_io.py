"""Recover a transient sharing violation without deleting the last good file."""
from pathlib import Path

from conn_assoc.phase23_core import replace_with_retry


def test_atomic_checkpoint_replace_retries_transient_permission_error(tmp_path,monkeypatch):
    target=tmp_path/'latest.pt';target.write_bytes(b'last good checkpoint')
    temp=tmp_path/'latest.tmp';temp.write_bytes(b'new checkpoint')
    original=Path.replace;calls=[]
    def locked_once(self,destination):
        calls.append(1)
        if len(calls)==1:
            assert target.read_bytes()==b'last good checkpoint'
            raise PermissionError('simulated Windows sharing violation')
        return original(self,destination)
    monkeypatch.setattr(Path,'replace',locked_once)
    replace_with_retry(temp,target)
    assert len(calls)==2 and target.read_bytes()==b'new checkpoint'
    assert (tmp_path/'io_retries.jsonl').is_file()
