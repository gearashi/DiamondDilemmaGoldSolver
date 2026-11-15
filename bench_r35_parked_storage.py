"""Compare an immutable parked bank plus offset with standalone periodic and final files."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import zipfile

import numpy as np

from bench_r35_checkpoint_storage import states, io_written
from search_problem import SearchProblem


def compressed(path,fields):
    with path.open('wb') as f:
        with zipfile.ZipFile(f,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
            for name,values in fields.items():
                with z.open(name+'.npy','w',force_zip64=True) as out:
                    np.lib.format.write_array(out,np.asanyarray(values),allow_pickle=False)
        f.flush()
        os.fsync(f.fileno())


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def suffix(key,value,offset):
    if key=='exact_paused_prefix_codes' or value.ndim==1:
        return value[offset:]
    return value[:,offset:]


def verify(active,reference):
    with np.load(active,allow_pickle=False) as disk:
        fields={k:disk[k] for k in disk.files}
    if 'bank_digest' in fields:
        sha=str(fields.pop('bank_digest'))
        offset=int(fields.pop('bank_offset'))
        obj=active.parent/(sha+'.npz')
        assert digest(obj)==sha
        with np.load(obj,allow_pickle=False) as bank:
            fields.update({k:suffix(k,bank[k],offset) for k in bank.files})
    assert fields.keys()==reference.keys()
    assert all(v.dtype==fields[k].dtype and np.array_equal(v,fields[k]) for k,v in reference.items())


def compare(paths,folder,portable_final):
    full=folder/'flat.npz'
    active=folder/'active.npz'
    base=identity=None
    rows=[]
    for i,src in enumerate(paths):
        with np.load(src,allow_pickle=False) as f:
            fields={k:f[k] for k in f.files}
        before=io_written()
        compressed(full,fields)
        after=io_written()
        ordinary=after-before if before is not None else full.stat().st_size
        verify(full,fields)
        paused={k:v for k,v in fields.items() if k.startswith('exact_paused_')}
        if not len(paused['exact_paused_job_ids']) or (portable_final and i==len(paths)-1):
            hot=fields
            joined=False
            offset=0
        else:
            offset=0 if base is None else len(base['exact_paused_job_ids'])-len(paused['exact_paused_job_ids'])
            joined=(base is not None and offset>=0
                and all(np.array_equal(v,suffix(k,base[k],offset)) for k,v in paused.items()))
            hot={k:v for k,v in fields.items() if k not in paused}
        prior=io_written()
        bank_bytes=0
        if hot is not fields:
            if not joined:
                tmp=folder/'newbank.npz'
                compressed(tmp,paused)
                identity=digest(tmp)
                tmp.replace(folder/(identity+'.npz'))
                bank_bytes=(folder/(identity+'.npz')).stat().st_size
                base,offset=paused,0
            hot.update(bank_digest=np.array(identity),bank_offset=np.int64(offset))
        compressed(active,hot)
        after=io_written()
        written=after-prior if prior is not None else active.stat().st_size+bank_bytes
        verify(active,fields)
        rows.append(dict(ordinal=i,flat_write=ordinary,split_write=written,
                 active_size=active.stat().st_size,new_bank_bytes=bank_bytes,reused=joined,
                 offset=offset,paused=len(paused['exact_paused_job_ids']),standalone=hot is fields))
    return rows


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--tiles',type=Path,required=True)
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args()
    original=SearchProblem(json.loads(args.tiles.read_bytes()))
    out={}
    with TemporaryDirectory(dir=Path(__file__).parent/'audit/r35') as tmp:
        for initial,active in ((2048,1),(8192,128),(128,128)):
            base=Path(tmp)/f'{initial}-{active}'
            base.mkdir()
            files=states(original,base,active,initial,initial_lanes=initial if initial!=active else None)
            for final in (False,True):
                work=base/str(final)
                work.mkdir()
                key=f'{initial}-to-{active}/portable-final-{final}'
                out[key]=compare(files,work,final)
    args.output.write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
