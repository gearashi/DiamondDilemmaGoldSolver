"""Bounded signed-loss positive searches around independent physically priced phases."""
import argparse
import json
import random
from pathlib import Path
from time import monotonic
from bench_r35_loss_proof import original, read_json, reference_groups
from r35_loss_witness import physical_price
from r35_integer_sat import InventorySAT
from io_utils import atomic_json


def solve(tiles, seedfile, seconds):
    problem,ref,packed,req,out=original(tiles)
    _,_,groups=reference_groups(problem,read_json('docs/R29_INCUMBENT.json')['codes'])
    extras=[g for g in groups if g not in packed]
    start=int(read_json(seedfile)['phase'],16)
    checked=physical_price(problem.tiles,ref,groups,req,start)[0]
    target=checked-1
    deadline=monotonic()+seconds
    native=InventorySAT(problem,packed,req,out,target,excess=True,additional_groups=extras,projected=True)
    donor=InventorySAT(problem,packed,req,out,checked,excess=True,additional_groups=extras,projected=True)
    rng=random.Random(3519)
    anchors=[start]
    positives=[]
    stats={'target_queries':0,'target_unsat_slices':0,'new_checked_nearby':0,'donor_queries':0}
    try:
        while monotonic()<deadline:
            anchor=anchors[len(positives)%len(anchors)] if len(anchors)==1 else rng.choice(anchors)
            for count in (96,72,48,32,20,8):
                if deadline<=monotonic():
                    break
                indices=rng.sample(range(1,problem.n),count)
                pins=[i+1 if anchor>>i&1 else -i-1 for i in indices]
                stats['target_queries']+=1
                verdict=native.decide(min(.4,max(.001,deadline-monotonic())),pins)
                if verdict is True:
                    phase=native.phase()
                    cost,failed=physical_price(problem.tiles,ref,groups,req,phase)
                    assert cost<=target
                    return dict(status='SAT',target=target,cost=cost,phase=hex(phase),stats=stats)
                if verdict is False:
                    stats['target_unsat_slices']+=1
            if deadline<=monotonic():
                break
            # Seek another locally feasible donor without ever excluding an unpinned target.
            stats['donor_queries']+=1
            indices=rng.sample(range(1,problem.n),60)
            pins=[i+1 if anchor>>i&1 else -i-1 for i in indices]
            donor.solver.add_clause([-(i+1) if anchor>>i&1 else i+1 for i in range(problem.n)])
            if donor.decide(min(.9,max(.001,deadline-monotonic())),pins):
                phase=donor.phase()
                cost,_=physical_price(problem.tiles,ref,groups,req,phase)
                assert cost<=checked
                if phase not in anchors:
                    anchors.append(phase)
                    stats['new_checked_nearby']+=1
                if cost<=target:
                    return dict(status='SAT',target=target,cost=cost,phase=hex(phase),stats=stats)
            positives.append(len(anchors))
        return dict(status='UNKNOWN',cap=target,seconds=seconds,stats=stats)
    finally:
        native.close()
        donor.close()


if __name__=='__main__':
    c=argparse.ArgumentParser(description=__doc__)
    c.add_argument('--tiles',default='../data/tiles.json')
    c.add_argument('--seedfile',type=Path,required=True)
    c.add_argument('--seconds',type=float,default=120)
    c.add_argument('--output',type=Path,required=True)
    args=c.parse_args()
    result=solve(args.tiles,args.seedfile,args.seconds)
    atomic_json(args.output,result)
    print(result)
