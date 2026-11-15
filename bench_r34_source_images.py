"""Measure repeated exact-run source images and compare publication strategies."""
import argparse
import ast
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from unittest.mock import patch

import exact_source_evidence
from source_image_storage import source_image_writer


def archive_counts(history):
    images={}
    byte_count=file_count=runs=0
    for run in history.iterdir():
        if not run.is_dir():
            continue
        files=[f for f in run.iterdir() if f.is_file() and f.suffix in ('.py','.c','.cu','.wgsl')]
        if files:
            runs+=1
        for f in files:
            data=f.read_bytes()
            images[hashlib.sha256(data).digest()]=len(data)
            byte_count+=len(data)
            file_count+=1
    unique_bytes=sum(images.values())
    return dict(runs=runs,files=file_count,bytes=byte_count,distinct=len(images),unique_bytes=unique_bytes,
                repeated_bytes=byte_count-unique_bytes)


def compare(root,work):
    script=ast.parse((root/'constraint_search.py').read_bytes())
    explicit=next(ast.literal_eval(n.value) for n in ast.walk(script) if isinstance(n,ast.Assign)
                  and any(isinstance(t,ast.Name) and t.id=='source_names' for t in n.targets))
    reference=None
    result=[]
    with TemporaryDirectory(dir=work) as folder:
        for number,trial in enumerate((False,True,True,False,False,True)):
            dest=Path(folder)/str(number)
            dest.mkdir()
            stats=Counter()
            @contextmanager
            def measured(destination):
                with source_image_writer(destination) as regular:
                    def write(source,data):
                        if trial:
                            cloned=regular(source,data)
                        else:
                            (destination/source.name).write_bytes(data)
                            cloned=False
                        stats['cloned_files' if cloned else 'written_files']+=1
                        stats['cloned_bytes' if cloned else 'written_bytes']+=len(data)
                    yield write
            start=time.perf_counter()
            with patch.object(exact_source_evidence,'source_image_writer',measured):
                hashes=exact_source_evidence.snapshot_search_sources(root,dest,explicit)
            elapsed=time.perf_counter()-start
            independently={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in dest.iterdir()}
            assert independently==hashes and (reference is None or reference==hashes)
            reference=hashes
            result.append(dict(trial=trial,seconds=elapsed,**stats))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--history',type=Path,required=True,help='A parent directory of prior run archives.')
    p.add_argument('--work',type=Path,required=True,help='Existing scratch directory on the destination volume.')
    args=p.parse_args()
    print(json.dumps(dict(historical=archive_counts(args.history),
                          current=compare(Path(__file__).resolve().parent,args.work)),indent=2))
