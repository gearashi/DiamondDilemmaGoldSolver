"""Independent signed neighbor phases and geographically checked fixed-phase residuals."""
from collections import defaultdict
from itertools import combinations
from pathlib import Path
import argparse
import json
from random import Random

from bench_r35_geographic_loss_cores import price_and_separate, connected_pose_count
from bench_r35_loss_boundary344 import problem_scope
from r28_geo_sewing import PosePacking
from r31_gold_connectivity import PhaseInterface
from r35_geographic_replay import geographic_departures
from r35_loss_witness import physical_price
from r35_signed_conditions import SignedConditions


def four_flip_neighbors(problem, required, phase, samples):
    interface=PhaseInterface(problem)
    integer=[]
    parity=[]
    for i,(low,high) in enumerate(interface.rows):
        sign=-1 if phase>>i&1 else 1
        changes=[sign*(b-a) for a,b in zip(low[:-1],high[:-1])]
        integer.append(tuple(v for v,mod in zip(changes,interface.mod[:-1]) if not mod))
        mask=sum((v%2)<<j for j,(v,mod) in enumerate(zip(changes,interface.mod[:-1])) if mod)
        for j,(a,b) in enumerate(sorted(required),len(interface.mod)):
            if i in (a//3,b//3): mask^=1<<j
        parity.append(mask)
    pair=[]
    lookup=defaultdict(list)
    for a,b in combinations(range(1,problem.n),2):
        sig=tuple(x+y for x,y in zip(integer[a],integer[b]))
        p=parity[a]^parity[b]
        lookup[sig,p].append((a,b))
        pair.append((a,b,sig,p))
    phases=[]
    for a,b,sig,p in pair:
        for c,d in lookup.get((tuple(-v for v in sig),p),()):
            if b<c: phases.append(phase^(1<<a)^(1<<b)^(1<<c)^(1<<d))
    Random(35081).shuffle(phases)
    return len(phases),phases[:samples]


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--tiles',required=True);cli.add_argument('--output',required=True,type=Path)
    cli.add_argument('--samples',type=int,default=32);cli.add_argument('--separate',type=int,default=4)
    cli.add_argument('--disconnected',action='store_true')
    args=cli.parse_args()
    p,ref,groups,req=problem_scope(args.tiles)
    initial=json.loads((Path(__file__).parent/'docs/R35_GEOGRAPHIC_LOSS_CORES.json').read_bytes())
    full=[frozenset(map(tuple,x)) for x in initial['geographic_cores']]
    known=geographic_departures(p,ref,req,full)
    pack=PosePacking(p)
    evidence={'puzzle':p.fingerprint,'phase_donors':[],'held_out':[],'separations':[]}
    def save():
        evidence['geographic_cores']=[sorted(g) for g in full]
        args.output.write_text(json.dumps(evidence,indent=2)+'\n',encoding='utf8')
    phases=set()
    for entry in sorted(initial['reports'],key=lambda x:-x['cap']):
        base=int(entry['phase'],16)
        count,rows=four_flip_neighbors(p,req,base,args.samples)
        phases.update(rows);evidence['phase_donors'].append(dict(phase=hex(base),eligible=count,sampled=len(rows)))
    euler=SignedConditions(p,euler=True);mates=SignedConditions(p,mates=True)
    for ph in sorted(phases):
        costs=[physical_price(p.tiles,ref,gs,req,ph) for gs in [groups[:13],[*groups[:13],*known],groups,[*groups,*known]]]
        assert all(v is not None for v in costs)
        a,b,c,d=[x[0] for x in costs]; assert a<=b and c<=d
        evidence['held_out'].append(dict(phase=hex(ph),base13=a,geo13=b,base344=c,geo344=d,
                                          euler=euler.accepts(ph),mates=mates.accepts(ph)))
    save()
    # Compare the closest previously unseen independent Hamming-four donors
    # at their new actual boundary, allowing every minimizing loss choice.
    ordered=sorted(evidence['held_out'],key=lambda row:(row['geo344'],row['phase']))
    for row in ordered[:args.separate]:
        phase=int(row['phase'],16); price=physical_price(p.tiles,ref,[*groups,*(g-req for g in full)],req,phase)[0]
        result=price_and_separate(p,ref,groups,req,phase,price,pack,full,disconnected=args.disconnected)
        if args.disconnected:
            from r35_geographic_packing_check import check_packing
            assert all(check_packing(p,g)['feasible'] is False for g in full)
        else:
            assert all(connected_pose_count(p,g)==0 for g in full)
        evidence['separations'].append(dict(phase=row['phase'],before=price,**result))
        save();print(row['phase'],price,result['status'],result.get('minimum'),len(full),flush=True)


if __name__=='__main__':main()
