"""Reproduce the original-puzzle checkpoint and CUDA comparisons recorded in round 34."""
import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
from unittest.mock import patch

import numpy as np

from r27_incumbent import BoardModel
from r28_exchange import ExchangeBound, read_json
from r29_exchange_clusters import StarClusters
from r30_bounds import defect_cores
from r30_multicenter import run
from r31_sampling import sample_archive
from search_problem import SearchProblem


def sampling(problem, backend):
    with tempfile.TemporaryDirectory() as temp:
        hot, cold = (Path(temp)/(name+'.sqlite') for name in ('hot','cold'))
        opts = dict(backend=backend,seed=122,replicas=128)
        prefix = sample_archive(problem,hot,dispatches=3,**opts)
        with patch('gpu_sampling_resume_round25.SamplerSession',side_effect=AssertionError('allocated on exact replay')):
            replay = sample_archive(problem,hot,dispatches=3,**opts)
        assert prefix['rows']==replay['rows'] and replay['new_proposals']==0 and replay['reused_dispatches']==3
        extension=sample_archive(problem,hot,dispatches=9,**opts)
        reference=sample_archive(problem,cold,dispatches=9,**opts)
        def state(path):
            with closing(sqlite3.connect(path)) as conn:
                return json.loads(conn.execute('SELECT state FROM r31_sampler WHERE launches=9').fetchone()[0])
        a,b=state(hot),state(cold)
        for key in ('native','archive','last_collected','solution','experiment_stats'):
            assert a[key]==b[key],key
        assert extension['rows']==reference['rows']
        return dict(backend=backend,device=a['device'],archive=len(replay['rows']),replay_sessions=0,
                    replay_proposals=replay['new_proposals'],extension_proposals=extension['new_proposals'],
                    cold_proposals=reference['new_proposals'])


def cuda_filters(problem):
    import cupy as cp
    from r30_gpu_filters import CudaPageFilters, contact_counts, core_passes
    model=BoardModel(problem)
    boards=[[3*t for t in item['pieces']] for item in read_json(Path(__file__).parent/'docs/R29_EXCHANGE_CENTERS.json')['centers']]
    bounds=[ExchangeBound(model,board) for board in boards]
    cores=[tuple(defect_cores(bound,208)) for bound in bounds]
    close=StarClusters(bounds[0]).close
    rng=np.random.default_rng(35)
    pages=[np.array([rng.choice(model.n,7,replace=False) for _ in range(1024)],np.int32) for _ in range(8)]
    for c,g in ((cores,close),([[],[0],[1<<(model.n-1)]],None),(None,close)):
        trial=CudaPageFilters(model.n,c,g)
        for page in [pages[0][:0],pages[0][:1],*pages]:
            a,b=trial.evaluate(page)
            if c is not None: np.testing.assert_array_equal(a,core_passes(page,c,model.n))
            if g is not None: np.testing.assert_array_equal(b,contact_counts(page,g))
    def measure(f):
        sent,received=[],[]
        send,receive=cp.asarray,cp.asnumpy
        def put(value,*args,**kw):
            device=send(value,*args,**kw)
            if not isinstance(value,cp.ndarray): sent.append(device.nbytes)
            return device
        def get(value,*args,**kw):
            received.append(value.nbytes)
            return receive(value,*args,**kw)
        with patch.object(cp,'asarray',side_effect=put),patch.object(cp,'asnumpy',side_effect=get):
            f()
        return dict(uploads=len(sent),downloads=len(received),uploaded_bytes=sum(sent),downloaded_bytes=sum(received))
    def separate():
        for page in pages:
            core_passes(page,cores,model.n,'cuda')
            contact_counts(page,close,'cuda')
    def fused():
        filters=CudaPageFilters(model.n,cores,close)
        for page in pages: filters.evaluate(page)
    with tempfile.TemporaryDirectory() as temp:
        opts=dict(families=((2,2,3),),start=307200000000,stop=307200000256,page=64,
                  filters=('contact','core','connected','star','family'))
        scalar=run(problem,boards,Path(temp)/'cpu.sqlite',backend='cpu',**opts)
        device=run(problem,boards,Path(temp)/'gpu.sqlite',backend='cuda',**opts)
        assert scalar==device
    return dict(control=measure(separate),trial=measure(fused),work=scalar['work'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment',choices=('cuda','webgpu','filters'))
    parser.add_argument('--tiles',default='data/tiles.json')
    args=parser.parse_args()
    problem=SearchProblem(read_json(args.tiles))
    print(json.dumps(cuda_filters(problem) if args.experiment=='filters' else sampling(problem,args.experiment)))
