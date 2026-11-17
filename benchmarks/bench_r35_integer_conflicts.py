"""Learn physical compulsory departures under an unrestricted loss budget; recheck each on a second solver."""
import argparse
from collections import Counter
from pathlib import Path
import json
import random
from time import monotonic

from bench_r35_loss_proof import original,read_json,reference_groups
from r35_integer_sat import InventorySAT
from io_utils import atomic_json


def learn(tiles,seconds):
    p,ref,packed,req,out=original(tiles)
    _,_,groups=reference_groups(p,read_json('docs/R29_INCUMBENT.json')['codes'])
    extras=[g for g in groups if g not in packed]
    first=InventorySAT(p,packed,req,out,len(ref),additional_groups=extras)
    other=InventorySAT(p,packed,req,out,len(ref),additional_groups=extras,native='m22')
    idx={v:b for b,v in first.physical_losses.items()}
    rng=random.Random(3520)
    accepted=set()
    known=set(groups)
    deadline=monotonic()+seconds
    work=Counter()
    def decide(check,literals,allowance=.12):
        left=deadline-monotonic()
        return check.decide(min(left,allowance),literals) if left>0 else None
    try:
        for trial in range(192):
            if monotonic()>=deadline:break
            order=list(idx)
            rng.shuffle(order)
            if trial%2:
                # Let each known group have at least one departure. A fresh
                # contradiction must survive after puncturing the original 13.
                omit={first.physical_losses[rng.choice(sorted(group-req))] for group in groups}
                order=[v for v in order if v not in omit]
            for _ in range(28):
                answer=decide(first,[-v for v in order]);work['queries']+=1
                if answer is not False:break
                core=[-v for v in first.solver.get_core()]
                if not core:raise ArithmeticError('Unrestricted source has no phase.')
                rng.shuffle(core)
                for width in (12,4,1):
                    k=0
                    while k<len(core) and monotonic()<deadline:
                        smaller=core[:k]+core[k+width:]
                        work['shrinks']+=1
                        if smaller and decide(first,[-v for v in smaller],.025) is False:
                            core=[-v for v in first.solver.get_core()]
                            k=0
                        else:k+=width
                proof=frozenset(idx[v] for v in core)
                excluded=set(core)
                order=[v for v in order if v not in excluded]
                if any(g<=proof for g in known|accepted):continue
                replay=decide(other,[-other.physical_losses[b] for b in proof],.75)
                work['independent_replays']+=1
                if replay is True:raise AssertionError('Different native solver recovered a counterexample.')
                if replay is not False:continue
                accepted={g for g in accepted if not proof<g}
                accepted.add(proof)
                if len(accepted)%12==1:print('cores',len(accepted),'size',len(proof),flush=True)
    finally:
        first.close();other.close()
    return dict(seconds=seconds,work=dict(work),groups=[sorted(g) for g in sorted(accepted,key=lambda g:(len(g),sorted(g)))],
                known_sizes=sorted(map(len,known)),scope='unrestricted physical loss budget, signed integer inventories, all 13 original groups and eight mandatory bonds')


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--tiles',default='../data/tiles.json')
    cli.add_argument('--seconds',type=float,default=75)
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args()
    result=learn(args.tiles,args.seconds)
    atomic_json(args.output,result)
    print(len(result['groups']),result['work'])
