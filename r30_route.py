"""Resumable native exact joins of two physical cycles for soft seam targets."""
import argparse
from contextlib import closing
import ctypes
import hashlib
import json
import math
import sqlite3
import time
from pathlib import Path

import numpy as np

from build_native import _build_one,library_name
from r27_incumbent import BoardModel
from r28_exchange import ExchangeBound, domain_binding, read_json
from r28_exchange_rotations import RotationOracle
from r29_exchange_clusters import StarClusters
from r29_exchange_route import ROOT,rotation_optimum,physical_hash,best_swap_path
from r30_bounds import budget_upper, defect_packings
from r30_cycles import Predecessors
from search_problem import SearchProblem


def library(stem='r30_cycle_native'):
    if stem not in ('r30_cycle_native','r30_group_native'):raise ValueError('Unknown native library.')
    sources=(ROOT/'r30_cycle_native.c').read_bytes()
    if stem=='r30_group_native':sources+=(ROOT/'r30_group_native.c').read_bytes()
    digest=hashlib.sha256(sources).hexdigest()[:20]
    directory=ROOT/'.native'/digest
    path=directory/library_name(stem)
    if not path.exists():_build_one(stem=stem,output_dir=directory,manifest_name='cycle-build.json')
    dll=ctypes.CDLL(str(path))
    dll.cycle_abi.argtypes=[]
    dll.cycle_abi.restype=ctypes.c_int
    if dll.cycle_abi()!=1:raise RuntimeError("Native two-cycle ABI mismatch.")
    c=ctypes.c_int
    ptr=ctypes.c_void_p
    arr=lambda dtype:np.ctypeslib.ndpointer(dtype,flags='C_CONTIGUOUS')
    i,u,h=arr(np.int32),arr(np.uint8),arr(np.int16)
    dll.cycle_new.argtypes=[c,c,i,i,u,u,u,h,c]
    dll.cycle_new.restype=ptr
    dll.cycle_page.argtypes=[ptr,c,c,c,c,c,arr(np.uint64),i,c]
    dll.cycle_page.restype=c
    dll.cycle_free.argtypes=[ptr]
    dll.cycle_free.restype=None
    return dll


class NativeCycles:
    def __init__(self,bound,target,family,*,component=False):
        self.family=tuple(family)
        if self.family not in ((4,2),(3,3)) or not 1<=bound.n<=240:
            raise ValueError('Unsupported two-cycle family or board dimension.')
        self.n=n=bound.n
        import itertools
        self.dll=library('r30_group_native') if component else library()
        # A star is a maximum of 3 binary comparisons. Varying a neighbor
        # changes the maximum by at most one; varying the central identity by
        # at most three. The mixed difference of two groups is bounded by the
        # sum of pairwise mixed oscillations, even with other changes present.
        nb=np.ascontiguousarray(bound.neighbors[:n],dtype=np.int32)
        clusters=StarClusters(bound)
        inter=np.zeros((n,n),np.int16)
        for c in range(n):
            weights={c:3}
            for other in nb[c]:
                if other<n:weights[other]=weights.get(other,0)+1
            for (a,wa),(b,wb) in itertools.combinations(weights.items(),2):
                inter[a,b]+=2*min(wa,wb)
                inter[b,a]+=2*min(wa,wb)
        masks=bound.support[bound.masks[bound.tiles][...,None],bound.pieces]
        masks=(masks*np.array([1,2,4])[None,:,None,None]).sum(1)
        free=np.zeros(n,np.uint8);free[bound.model.free]=1
        self.args=(nb,np.ascontiguousarray(clusters.delta,np.int32),np.ascontiguousarray(masks,np.uint8),
                   np.ascontiguousarray(bound.base[:n],np.uint8),free,inter)
        self.ptr=self.dll.cycle_new(n,target,*self.args,self.family[1])
        if not self.ptr:raise MemoryError('Could not allocate cycle-join catalogue.')

    def page(self,start,stop,certified,cap=2048):
        if (not self.ptr or any(type(value) is not int for value in (start,stop,cap))
                or not 0<=start<=stop<=self.n**2 or not 1<=cap<=2**27):
            raise ValueError('Invalid native cycle page.')
        counters=np.zeros(4,np.uint64)
        out=np.empty((cap,6),np.int32)
        n=self.dll.cycle_page(self.ptr,start,stop,*self.family,int(certified),counters,out,cap)
        if n==-1:
            if stop-start==1:return self.page(start,stop,certified,cap*2)
            mid=(start+stop)//2
            a,ar=self.page(start,mid,certified,cap)
            b,br=self.page(mid,stop,certified,cap)
            return {k:a[k]+b[k] for k in a},np.concatenate((ar,br))
        if n<0:raise ValueError('Native key validation failed')
        return dict(zip(('outer','joins','star_queries','survivors'),map(int,counters))),out[:n]

    def close(self):
        if self.ptr:self.dll.cycle_free(self.ptr);self.ptr=None


def search(problem,codes,target,database,family,*,batch=16,seconds=0,max_pages=None,
           stop_on_hit=False,predecessors=True,deep=False):
    family=tuple(family)
    if family not in ((4,2),(3,3)):raise ValueError('Expected a directed four-cycle and swap or two triples.')
    if batch<1 or seconds<0 or not 0<=target<=len(problem.edges):raise ValueError('Invalid budget or target')
    model=BoardModel(problem)
    bound=ExchangeBound(model,codes)
    cert=Predecessors(bound,target)
    start=time.monotonic()
    binding=domain_binding(model)
    revision=hashlib.sha256((ROOT/'r30_cycle_native.c').read_bytes()).hexdigest()[:16]
    key=json.dumps([2,revision,binding,physical_hash(bound.tiles),family,target,predecessors,deep])
    oracle=RotationOracle(model)
    engine=None
    ended=bound.n**2
    empty=dict(cursor=0,outer=0,joins=0,star_queries=0,survivors=0,undo=0,exact=0,cached=0,hits=0,best=None)
    pages=0
    ceiling=None
    with closing(sqlite3.connect(database,timeout=120)) as db:
        db.execute('CREATE TABLE IF NOT EXISTS cycle_pages (key TEXT PRIMARY KEY, data TEXT)')
        db.execute('CREATE TABLE IF NOT EXISTS optima (key TEXT PRIMARY KEY,result TEXT NOT NULL)')
        try:
            while True:
                db.execute('BEGIN IMMEDIATE')
                row=db.execute('SELECT data FROM cycle_pages WHERE key=?',(key,)).fetchone()
                state=json.loads(row[0]) if row else empty.copy()
                pos=state['cursor']
                if (pos>=ended or (seconds and time.monotonic()-start>=seconds) or
                        (max_pages is not None and pages>=max_pages) or (stop_on_hit and state['hits'])):
                    db.commit();break
                if ceiling is None:
                    ceiling=budget_upper(bound,defect_packings(bound),4)
                    if ceiling<target:
                        state.update(cursor=ended,budget_upper=ceiling)
                        db.execute('INSERT OR REPLACE INTO cycle_pages VALUES (?,?)',(key,json.dumps(state)))
                        db.commit()
                        break
                if engine is None:engine=NativeCycles(bound,target,family)
                stop=min(ended,pos+batch)
                stats,rows=engine.page(pos,stop,cert.certified and predecessors)
                for k,v in stats.items():state[k]+=v
                # Witness-first within this page, while retaining one atomic proof cursor.
                order=tuple(list(range(1,family[0]))+[0]+list(range(family[0]+1,6))+[family[0]])
                if len(rows):
                    scores=bound.bounds(rows,order)
                    if np.any(scores<target):raise ArithmeticError('Native survivor disagrees with portable star.')
                    rows=rows[np.argsort(-scores,kind='stable')]
                for cells in rows:
                    cycles=(tuple(map(int,cells[:family[0]])),tuple(map(int,cells[family[0]:])))
                    if predecessors and cert.reject(cycles,deep=deep):
                        state['undo']+=1;continue
                    seed=bound.witness_seed(cells,order)
                    if seed is None:continue
                    k=binding+':'+','.join(map(str,seed//3))
                    old=db.execute('SELECT result FROM optima WHERE key=?',(k,)).fetchone()
                    if old:
                        state['cached']+=1;item=json.loads(old[0])
                    else:
                        item=rotation_optimum(bound,oracle,seed)
                        db.execute('INSERT INTO optima VALUES (?,?)',(k,json.dumps(item)))
                        state['exact']+=1
                    if item['score']>=target:
                        state['hits']+=1
                        if state['best'] is None or item['score']>state['best']['score']:state['best']=item
                state['cursor']=stop
                db.execute('INSERT OR REPLACE INTO cycle_pages VALUES (?,?)',(key,json.dumps(state)))
                db.commit();pages+=1
        finally:
            if engine is not None:engine.close()
    return dict(state,scope=family,total=math.comb(len(model.free),6)*(90 if family==(4,2) else 40),
                target=target,center=physical_hash(bound.tiles),complete=state['cursor']==ended,
                end=ended,elapsed=time.monotonic()-start)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiles',default='data/tiles.json')
    parser.add_argument('--incumbent',default='docs/R29_INCUMBENT.json')
    parser.add_argument('--database',required=True)
    parser.add_argument('--family',choices=['42','33'],default='33')
    parser.add_argument('--target',type=int,default=208)
    parser.add_argument('--batch',type=int,default=16)
    parser.add_argument('--seconds',type=float,default=0)
    parser.add_argument('--deep',action='store_true')
    parser.add_argument('--recheck',action='store_true',help='Bypass the shipped scope certificate and use the requested checkpoint.')
    parser.add_argument('--stop-on-hit',action='store_true')
    parser.add_argument('--path',action='store_true')
    a=parser.parse_args()
    p=SearchProblem(read_json(a.tiles));codes=read_json(a.incumbent)['codes']
    family=tuple(map(int,a.family))
    model=BoardModel(p)
    if not 0<=a.target<=len(p.edges) or a.batch<1 or a.seconds<0:
        parser.error('Invalid target or run budget.')
    physical=tuple(model.legal_board(codes)//3)
    from r30_multicenter import released_six
    if not a.recheck and released_six(domain_binding(model),[physical],(tuple(sorted(family)),),a.target):
        r=dict(scope=family,center=physical_hash(physical),target=a.target,complete=True,best=None,hits=0,
               total=math.comb(len(model.free),6)*(90 if family==(4,2) else 40),
               source='docs/R30_ROUTES_RESULTS.json',star_queries_this_run=0)
    else:
        r=search(p,codes,a.target,a.database,family,batch=a.batch,seconds=a.seconds,
                 stop_on_hit=a.stop_on_hit,deep=a.deep)
    if a.path and r['best']:r['path']=best_swap_path(p,codes,r['best']['codes'])
    print(json.dumps(r))


if __name__=='__main__':main()
