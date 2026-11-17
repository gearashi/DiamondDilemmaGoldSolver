"""Find low-loss signed witnesses by exact neutral exchanges and new sampled six-tile exchanges."""
import argparse
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path
from random import Random
from time import monotonic

from bench_r35_loss_boundary344 import problem_scope
from r31_gold_connectivity import PhaseInterface
from r35_cover_search import reduced_cover
from r35_loss_witness import raw_sides,physical_price


def exchanges(interface,req,phase,rng,samples):
    signatures=[]
    for i,(lo,hi) in enumerate(interface.rows):
        sign=-1 if phase>>i&1 else 1
        integer=parity=0
        for k,(a,b,mod) in enumerate(zip(lo[:-1],hi[:-1],interface.mod[:-1])):
            change=sign*(b-a)
            if mod:parity|=(change%2)<<k
            else:integer+=change<<(7*k)
        for k,(a,b) in enumerate(sorted(req),len(interface.mod)):
            if i in (a//3,b//3):parity^=1<<k
        signatures.append((integer,parity))
    tree=defaultdict(list)
    n=len(signatures)
    p2=[]
    for a,b in combinations(range(1,n),2):
        v=signatures[a][0]+signatures[b][0];y=signatures[a][1]^signatures[b][1]
        mask=(1<<a)|(1<<b);tree[v,y].append((a,b,mask));p2.append((b,v,y,mask))
        if v==0 and not y:yield phase^mask
    for a,(v,y) in enumerate(signatures[1:],1):
        for b,c,mask in tree[-v,y]:
            if a<b:yield phase^(1<<a)^mask
    for b,v,y,mask in p2:
        for c,d,other in tree[-v,y]:
            if b<c:yield phase^mask^other
    # Join actual three-tile integer and parity changes, allowing an exchange
    # even when none of its two-, three- or four-tile subsets is neutral.
    triples=defaultdict(list)
    exhaustive=samples==0
    triples_to_visit=combinations(range(1,n),3) if exhaustive else (sorted(rng.sample(range(1,n),3)) for _ in range(samples))
    for a,b,c in triples_to_visit:
        k=signatures[a][0]+signatures[b][0]+signatures[c][0]
        val=signatures[a][1]^signatures[b][1]^signatures[c][1]
        mask=(1<<a)|(1<<b)|(1<<c)
        for x,y,pair_mask in tree[-k,val]:
            if y<a:yield phase^mask^pair_mask
        for old in triples[-k,val]:
            if (old.bit_length()<=a if exhaustive else not old&mask):yield phase^old^mask
        triples[k,val].append(mask)


def run(tiles,geo,paths,out,seconds,samples,seed_phase=(),max_centers=0):
    p,ref,gs,req=problem_scope(tiles)
    read=lambda f,k:[frozenset(map(tuple,v))-req for v in json.loads(f.read_bytes())[k]]
    groups=[*gs[:13],*read(geo,'geographic_cores'),
            *[frozenset(map(tuple,g))-req for g in json.loads(paths.read_bytes())['edge_perfect_cores'] if len(g)<=13]]
    labels=raw_sides(p.tiles)
    rules=[(pair,(1<<(u//3))^(1<<(v//3)),labels[u][1]^labels[v][1]^(len(labels[u][0])%2))
           for pair in sorted(ref|req) for u,v in [pair]]
    interface=PhaseInterface(p);rng=Random(8135)
    active={};seen=set();expanded=set();leader=241;records=[];total=0
    start=monotonic()
    def save():
        out.write_text(json.dumps(dict(seconds=monotonic()-start,expanded=len(expanded),considered=total,
            phases=len(seen),best=leader,records=records),indent=2)+'\n',encoding='utf8')
    def score(phase):
        nonlocal total,leader
        if phase in seen:return
        total+=1;seen.add(phase)
        failed={b for b,mask,rhs in rules if ((mask&phase).bit_count()%2)!=rhs}
        assert not failed&req
        forced=failed&ref
        remaining=[g-req for g in groups if not g&failed]
        # Every member selected from a pairwise disjoint family incurs one loss.
        used=set();lower=len(forced)
        for q in sorted(remaining,key=len):
            if not used&q:lower+=1;used|=q
        if lower>leader+3:return
        lost=forced|reduced_cover(remaining)
        cost=len(lost)
        if cost<=leader+3:active[phase]=cost
        if cost<leader:
            value=physical_price(p.tiles,ref,groups,req,phase)
            if value is None or value[0]!=cost:raise ArithmeticError('Raw signed replay rejected this proposed phase.')
            leader=cost;records.append(dict(cost=cost,phase=hex(phase),lost=sorted(lost)));save()
    for phase in (seed_phase or [0xce55aaa6e89109fca073547e8c580ba32e0e5fa2,0xd237aba7e91101e88041413e8c589902021e5126]):score(phase)
    while active and monotonic()-start<seconds and (not max_centers or len(expanded)<max_centers):
        phase=min(active,key=lambda ph:(active[ph],ph));active.pop(phase);expanded.add(phase)
        for other in exchanges(interface,req,phase,rng,samples):score(other)
        save()


if __name__=='__main__':
    cmd=argparse.ArgumentParser(description=__doc__)
    for x in ('tiles','geo','paths','out'):cmd.add_argument('--'+x,required=True,type=Path)
    cmd.add_argument('--seconds',default=240,type=int);cmd.add_argument('--samples',default=60000,type=int)
    cmd.add_argument('--seed-phase',nargs='*',type=lambda x:int(x,0));cmd.add_argument('--max-centers',type=int,default=0)
    a=cmd.parse_args();run(**vars(a))
