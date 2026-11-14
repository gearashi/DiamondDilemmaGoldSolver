"""Equal-budget native GPU seeding experiment, including checkpoint trajectory."""
import argparse,json,time
from pathlib import Path
import numpy as np
from search_problem import SearchProblem
from gpu_sampling_resume_round25 import SamplerSession



class SeededSession(SamplerSession):
    """Benchmark-only fresh-lane treatment; restore preserves all saved lanes."""
    def __init__(self,problem,*,initial_codes=None,resume=None,**kwargs):
        import hashlib
        super().__init__(problem,**kwargs)
        self.initial_codes=None if initial_codes is None else list(initial_codes)
        if self.initial_codes is not None:
            from gpu_sampling import score_board
            if score_board(problem,self.initial_codes)['domain_violations']:
                raise ValueError('Seed must be legal.')
            self.config['initial_codes']=self.initial_codes.copy()
            for name in ('bench_r27_incumbent_gpu.py','r27_incumbent_seed.py'):
                self.config['sources'][name]=hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
        if resume is not None:
            self._restore(resume)

    def _ensure_engine(self):
        fresh=self.engine is None and self.native is None
        super()._ensure_engine()
        if fresh and self.initial_codes is not None:
            from r27_incumbent_seed import initialize_lanes
            try:
                initialize_lanes(self.problem,self.engine,self.initial_codes,self.config['seed'])
            except BaseException:
                self.poisoned=True
                raise


def run(problem,codes,backend,seed,seeded,replicas,intervals,split=False,flags=None,gold=False):
    start=time.perf_counter()
    kwargs=dict(backend=backend,seed=seed,target='gold',replicas=replicas,gold_aware=gold,
                initial_codes=codes if seeded else None,experiments=flags or {'replica_exchange':True,'move_tabu':True})
    session=SeededSession(problem,**kwargs)
    rows=[]
    first=None
    baseline=problem.validate(codes)
    best=(baseline['matched_edges'],-baseline['line_components'])
    best_codes=codes
    try:
        session._ensure_engine()
        for tick in range(intervals+1):
            if tick:
                session.advance(900,max_dispatches=1)
            session.collect()
            for row in session.archive:
                if row['domain_violations']:
                    continue
                check=problem.validate(row['codes'])
                assert check['matched_edges']==row['matched_edges']
                pair=(check['matched_edges'],-check['line_components'])
                if pair>best:
                    best=pair
                    best_codes=row['codes']
                if pair[0]>=208 and first is None:
                    first=tick*replicas*64
            rows.append([tick*replicas*64,session.report()['best_score'],best[0],-best[1]])
            if split and tick==intervals//2:
                saved=session.snapshot()
                session.close()
                session=SeededSession(problem,**kwargs,resume=saved)
                session._ensure_engine()
        import hashlib
        digest=hashlib.sha256(b''.join(x.tobytes() for x in session.engine.snapshot_arrays())).hexdigest()
        return dict(seed=seed,seeded=seeded,replicas=replicas,dispatches=intervals,proposals=intervals*replicas*64,
                    first208=first,score=best[0],components=-best[1],codes=best_codes,trajectory=rows,
                    native_hash=digest,backend=backend,device=session.device,seconds=time.perf_counter()-start)
    finally:
        session.close()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--backend',choices=['cuda','webgpu'],default='cuda')
    p.add_argument('--replicas',type=int,default=128)
    p.add_argument('--dispatches',type=int,default=32)
    p.add_argument('--seeds',type=int,default=4)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--restore',action='store_true')
    p.add_argument('--gold',action='store_true')
    p.add_argument('--from-polished',action='store_true')
    opts=p.parse_args()
    src=lambda name:json.loads(Path(name).read_text(encoding='utf-8-sig'))
    problem=SearchProblem(src('data/tiles.json'))
    codes=src('audit/r27/incumbent.json')['codes']
    if opts.from_polished:
        from r27_incumbent import polish
        codes=polish(problem,codes,rounds=8)['codes']
    flags={'replica_exchange':True,'move_tabu':True}
    if opts.gold:
        flags.update(lazy_gold_scoring=True,incremental_sampling=True)
    result=[]
    for seed in range(opts.seeds):
        for seeded in (False,True):
            item=run(problem,codes,opts.backend,seed,seeded,opts.replicas,opts.dispatches,flags=flags,gold=opts.gold)
            if opts.restore and seed==0 and seeded:
                replay=run(problem,codes,opts.backend,seed,seeded,opts.replicas,opts.dispatches,True,flags=flags,gold=opts.gold)
                assert (item['native_hash'],item['trajectory'])==(replay['native_hash'],replay['trajectory'])
                item['split_restore_equal']=True
            result.append(item)
            opts.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
            print(seed,seeded,item['score'],item['components'],item['first208'],flush=True)


if __name__=='__main__':
    main()

