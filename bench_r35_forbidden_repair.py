"""Locate creation and cleanup of forbidden physical mates during ordinary bonded repair."""
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
import argparse

import numpy as np

from io_utils import atomic_json
from r28_exchange import read_json
from r31_fresh import Engine,physical_bonds
from search_problem import SearchProblem


class AuditEngine(Engine):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.measured=Counter()
        self.examples=[]
        self.steps_with_collisions=set()

    def repair(self,board,rng,step,stats,prices):
        import r31_fresh
        m=self.model
        rules=self.assemblies
        def banned(values):
            return rules.forbidden & physical_bonds(m,values)
        def stage(name,fn,position):
            def observed(*args,**kwargs):
                values=args[position]
                old=banned(values)
                answer=fn(*args,**kwargs)
                result=answer[0] if isinstance(answer,tuple) else answer
                if result is None:
                    if name=='cleanup':
                        self.measured['cleanup_rollbacks']+=1
                        self.measured['rollback_lost_matches']+=int(m.matches(values).sum()-m.matches(board).sum())
                    return answer
                new=banned(result)
                if new-old:
                    self.steps_with_collisions.add(step)
                    self.measured[name+'_created_calls']+=1
                    self.measured[name+'_created_pairs']+=len(new-old)
                    if len(self.examples)<24:
                        self.examples.append(dict(stage=name,step=step,old=sorted([list(map(int,p)) for p in old]),
                                     new=sorted([list(map(int,p)) for p in new]),before=values.tolist(),after=result.tolist()))
                if name=='cleanup' and old:
                    gain=int(m.matches(result).sum()-m.matches(values).sum())
                    self.measured['cleanups']+=1
                    self.measured['cleanup_fewer_matches']+=int(gain<0)
                    self.measured['cleanup_gained_matches']+=max(0,gain)
                    self.measured['cleanup_lost_matches']+=max(0,-gain)
                return answer
            return observed
        with ExitStack() as stack:
            for target,name,label,pos in [(rules,'relocate','relocate',0),(r31_fresh,'simultaneous_swaps','swaps',1),
                                        (r31_fresh,'pack_cycles','circulation',1),(r31_fresh,'joint_repair','joint',1),
                                        (self.mate_rules if self.forbidden_repair in ('prices','guarded') else m,'block','block',0),(m,'orientations','rotations',0),
                                        (rules,'clear_forbidden','cleanup',0)]:
                original=getattr(target,name)
                stack.enter_context(patch.object(target,name,stage(label,original,pos)))
            data=super().repair(board,rng,step,stats,prices)
        if banned(np.asarray(data['codes'])):
            raise ValueError('Saved repair violates a forbidden mate.')
        return data


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tiles',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--policy',choices=('released','regional','single'),default='regional')
    ap.add_argument('--seeds',type=int,nargs='+',default=[3500,3501,3502,3503])
    ap.add_argument('--repair',choices=('project','prices','guarded'))
    ap.add_argument('--rounds',type=int,default=320)
    args=ap.parse_args()
    problem=SearchProblem(read_json(args.tiles))
    rows=[]
    for seed in args.seeds:
        engine=AuditEngine(problem,bonds=True,cycle_budget=3,exact_packing=args.policy=='regional',
                           regional_cycle_budget=2 if args.policy=='regional' else 0,
                           contact_group=1 if args.policy=='single' else 0,forbidden_repair=args.repair)
        row=engine.run(seed,rounds=args.rounds)
        result=dict(seed=seed,score=row['score'],steps=args.rounds,work=dict(engine.measured),
                    rounds_with_new_collisions=len(engine.steps_with_collisions),examples=engine.examples,
                    search_work=row['work'])
        rows.append(result)
        atomic_json(args.output,dict(input=problem.fingerprint,policy=args.policy,repair=args.repair,records=rows))
        print(seed,row['score'],dict(engine.measured),flush=True)


if __name__=='__main__':
    main()
