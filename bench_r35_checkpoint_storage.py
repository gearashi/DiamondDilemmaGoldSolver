"""Measure fsync-complete flat NPZ and incremental checkpoint costs on genuine classic native states."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from time import perf_counter
import zipfile
import zlib

import numpy as np


def io_written():
    if os.name != 'nt':
        return None
    import ctypes
    counters = ctypes.c_uint64 * 6
    kernel = ctypes.windll.kernel32
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.GetProcessIoCounters.argtypes = (ctypes.c_void_p, ctypes.POINTER(counters))
    kernel.GetProcessIoCounters.restype = ctypes.c_int
    data = counters()
    if not kernel.GetProcessIoCounters(kernel.GetCurrentProcess(), ctypes.byref(data)):
        return None
    return data[4]


def members(path):
    with zipfile.ZipFile(path) as source:
        for name in source.namelist():
            yield name,source.read(name)


def member_hashes(path):
    return {name:hashlib.sha256(data).hexdigest() for name,data in members(path)}


def flat(src,dst,level):
    before=io_written()
    start=perf_counter()
    with dst.open('wb') as f:
        method = zipfile.ZIP_DEFLATED if level else zipfile.ZIP_STORED
        with zipfile.ZipFile(f,'w',compression=method,compresslevel=level or None) as out:
            for name,data in members(src):
                out.writestr(name,data)
        f.flush()
        os.fsync(f.fileno())
    writes=io_written()
    return dict(file_bytes=dst.stat().st_size,write_bytes=writes-before if writes is not None else None,
                seconds=round(perf_counter()-start,5))


class Chunks:
    """Transactional comparison store. Each saved generation lists every byte in field order."""
    def __init__(self,path,chunk_size,level):
        self.db=sqlite3.connect(path)
        self.size=chunk_size
        self.level=level
        self.path=path
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('PRAGMA wal_autocheckpoint=0')
        self.db.executescript("""CREATE TABLE blob(hash BLOB PRIMARY KEY,data BLOB NOT NULL) WITHOUT ROWID;
            CREATE TABLE generation(number INTEGER PRIMARY KEY, manifest BLOB NOT NULL, digest BLOB NOT NULL);""")
        self.db.execute('PRAGMA wal_checkpoint(TRUNCATE)')

    def write(self,src,gen):
        start=perf_counter()
        prior=io_written()
        new=compressed=raw_bytes=parts=0
        manifest={}
        with self.db:
            for name,data in members(src):
                row=[]
                for i in range(0,len(data),self.size):
                    block=data[i:i+self.size]
                    key=hashlib.sha256(block).digest()
                    if not self.db.execute('SELECT 1 FROM blob WHERE hash=?',(key,)).fetchone():
                        payload=zlib.compress(block,self.level)
                        self.db.execute('INSERT INTO blob VALUES(?,?)',(key,payload))
                        compressed+=len(payload)
                        raw_bytes+=len(block)
                        new+=1
                    parts+=1
                    row.append(key.hex())
                manifest[name]=[len(data),row]
            body=json.dumps(manifest,separators=(',',':'),sort_keys=True).encode()
            packed=zlib.compress(body,1)
            self.db.execute('INSERT INTO generation VALUES(?,?,?)',(gen,packed,hashlib.sha256(packed).digest()))
        # Report both commits to WAL and publishing the committed pages into the main file.
        wal=self.path.with_name(self.path.name+'-wal').stat().st_size
        self.db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        end=io_written()
        return dict(unique_chunks=new,chunks=parts,new_raw_bytes=raw_bytes,new_compressed_bytes=compressed,
                    manifest=len(packed),wal_bytes=wal,file_bytes=self.path.stat().st_size,
                    write_bytes=end-prior if end is not None else None,seconds=round(perf_counter()-start,5))

    def verify(self,gen,hashes):
        raw,key=self.db.execute('SELECT manifest,digest FROM generation WHERE number=?',(gen,)).fetchone()
        assert hashlib.sha256(raw).digest()==key
        manifest=json.loads(zlib.decompress(raw))
        assert set(manifest)==set(hashes)
        for name,(length,refs) in manifest.items():
            digest=hashlib.sha256()
            count=0
            for ref in refs:
                block=zlib.decompress(self.db.execute('SELECT data FROM blob WHERE hash=?',(bytes.fromhex(ref),)).fetchone()[0])
                assert hashlib.sha256(block).hexdigest()==ref
                digest.update(block)
                count+=len(block)
            assert count==length and digest.hexdigest()==hashes[name]


def states(problem,base,lanes,used,*,initial_lanes=None):
    from dfs_gpu import DFS,fill_order
    from exact_frontier import plan_frontier
    from systematic_jobs import JobLedger
    order=fill_order(problem.neighbors)
    plan=plan_frontier(problem.masks,problem.neighbors,problem.sides,order,used)
    prefix,lengths=plan.prefixes[:used],plan.lengths[:used]
    def pool(count):
        return DFS(problem.masks,problem.neighbors,problem.sides,n=count,seed=35068,order=order,prefixes=[],gold_data=problem.tiles)
    gpu=pool(initial_lanes or lanes)
    ledger=JobLedger.empty(len(prefix),gpu.n)
    ledger.refill(gpu,prefix,lengths)
    if initial_lanes:
        gpu.step(32)
        ledger.refill(gpu,prefix,lengths)
        saved={**gpu.snapshot_checkpoint(),**ledger.fields()}
        gpu=pool(lanes)
        ledger=JobLedger.resume(gpu,saved,prefix,lengths)
        assert ledger.paused_count >= used//2,(ledger.paused_count,used)
    files=[]
    for gen,steps in enumerate((0,1,8,64,256)):
        if steps:
            gpu.step(steps)
            ledger.refill(gpu,prefix,lengths)
        saved=gpu.snapshot_checkpoint()
        for j in np.flatnonzero(saved['states']==1):
            codes=saved['boards'][:,j].astype(int).tolist()
            if problem.validate(codes)['valid']:
                (Path(__file__).parent/'audit/r35/checkpoint-gold.json').write_text(json.dumps(codes),encoding='utf8')
                raise SystemExit('Verified Gold candidate saved; search stopped.')
        saved.update(ledger.fields(),exact_generation=np.int64(gen))
        path=base/f'input-{gen}.npz'
        np.savez(path,**saved)
        files.append(path)
    return files


def run(paths,base,levels,widths):
    reference=[member_hashes(p) for p in paths]
    result={'flat':{},'chunks':{}}
    for level in levels:
        rows=[]
        for p,h in zip(paths,reference):
            dest=base/'flat.npz'
            rows.append(flat(p,dest,level))
            assert member_hashes(dest)==h
        result['flat'][str(level)]=rows
    for size in widths:
        path=base/f'chunks-{size}.sqlite'
        store=Chunks(path,size,1)
        try:
            result['chunks'][str(size)]=[store.write(src,gen) for gen,src in enumerate(paths)]
            for gen,h in enumerate(reference):
                store.verify(gen,h)
        finally:
            store.db.close()
    result['input_bytes']=[p.stat().st_size for p in paths]
    return result


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--tiles',required=True,type=Path)
    cli.add_argument('--installed',type=Path)
    cli.add_argument('--parked',action='store_true')
    cli.add_argument('--output',required=True,type=Path)
    args=cli.parse_args()
    from search_problem import SearchProblem
    p=SearchProblem(json.loads(args.tiles.read_bytes()))
    results={}
    location=Path(__file__).parent/'audit/r35'
    location.mkdir(exist_ok=True,parents=True)
    with TemporaryDirectory(dir=location,prefix='classic-benchmark-') as tmp:
        base=Path(tmp)
        if args.installed:
            results['installed']=run([args.installed],base,(0,1,6),(16384,65536,262144))
        elif args.parked:
            for initial,lanes in ((2048,1),(8192,128)):
                folder=base/f'{initial}-to-{lanes}'
                folder.mkdir()
                paths=states(p,folder,lanes,initial,initial_lanes=initial)
                result=run(paths,folder,(0,1,6),(4096,16384,65536,262144))
                with np.load(paths[0],allow_pickle=False) as f:
                    result['paused']=len(f['exact_paused_job_ids'])
                results[folder.name]=result
        else:
            for lanes,assigned in ((128,128),(2048,128),(2048,2048)):
                folder=base/f'{lanes}-{assigned}'
                folder.mkdir()
                results[folder.name]=run(states(p,folder,lanes,assigned),folder,(0,1,6),(4096,16384,65536,262144))
    args.output.write_text(json.dumps(results,indent=2)+'\n',encoding='utf8')
