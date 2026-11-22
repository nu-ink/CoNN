import csv
import hashlib
import json
import os
from pathlib import Path
import time

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''): h.update(b)
    return h.hexdigest()

def write(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    replace(tmp,path)

def replace(tmp,path):
    for attempt in range(10):
        try: os.replace(tmp,path); return
        except PermissionError as exc:
            with (Path(path).parent/'io_retries.jsonl').open('a') as f:
                f.write(json.dumps({'attempt':attempt+1,'path':str(path),'error':str(exc)})+'\n')
            if attempt==9: raise
            time.sleep(min(.5,.05*2**attempt))

def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))

def csv_write(path,rows):
    if not rows: return
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def csv_read(path):
    with Path(path).open(newline='',encoding='utf-8') as f: return list(csv.DictReader(f))
