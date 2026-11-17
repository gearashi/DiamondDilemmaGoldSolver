"""Measure extra retention bounds just below the published geographic limit."""
import argparse
import hashlib
from collections import Counter
import json
from pathlib import Path
from random import Random

from r27_structure_filters import domain_bits
from r28_exchange import read_json
from r32_gold_bonds import inventory
from r32_gold_loss import reference_groups
from r32_gold_watch import BondWatcher
from r35_alternative_retention import prune_alternatives
from r27_incumbent import assignment
from r27_structure_filters import bits
from collections import defaultdict
import numpy as np
from r35_geographic_loss import certified_reference
from search_problem import SearchProblem


def alternative_prices(domains,optional):
    """Every named optional bond contributes at most two units to a placement."""
    n=len(domains)
    doubled=np.zeros((n,3*n),dtype=np.int16)
    direct=defaultdict(set)
    for pair,choices in optional.items():
        one={code//3 for _,code in choices}
        if not one:continue
        charge=2 if len(one)==1 else 1
        for c,code in set(choices):
            doubled[c,code]+=charge
            direct[c,code].add(pair)
    return doubled,direct


def assignment_price(domains,prices):
    n=len(domains)
    costs=np.full((n,n),32767,dtype=np.int32)
    for c,row in enumerate(domains):
        for code in bits(row):
            t=code//3
            costs[c,t]=min(costs[c,t],prices[c,code])
    legal=costs!=32767
    if not legal.any(axis=0).all() or not legal.any(axis=1).all():
        return None,None,None
    independent=max(int(costs.min(axis=1).sum()),int(costs.min(axis=0).sum()))
    try:
        match=assignment(-costs,legal)
    except ValueError:
        return independent,None,None
    return independent,int(costs[np.arange(n),match].sum()),costs


def implied(p,domains,ref,required):
    watch=BondWatcher(p,());fixed=set(ref)&set(required);remaining={}
    for pair in sorted(ref-fixed):
        a,ap=watch._forced_poses(domains,*pair)
        b,bp=watch._forced_poses(domains,*reversed(pair))
        if a or b:fixed.add(pair)
        else:remaining[pair]=ap+bp
    return fixed,remaining


def screen(p,codes,seed,free,cover):
    from validator import reverse_mask
    rng=Random(seed)
    bad=[(a,b) for a,x,b,y in p.edges if int(p.masks[codes[a],x])!=reverse_mask(int(p.masks[codes[b],y]))]
    order=list(range(p.n));rng.shuffle(order);selected=set()
    if cover:
        while bad:
            freq=Counter(v for edge in bad for v in edge)
            c=min((v for v in freq if freq[v]==max(freq.values())),key=order.index)
            selected.add(c);bad=[e for e in bad if c not in e]
    selected.update(v for v in order[:free] if v not in selected)
    pool=sum(7 << (3*(codes[c]//3)) for c in selected)
    return [pool if c in selected else 1 << v for c,v in enumerate(codes)],len(selected)


def alternatives(p,ref,required,seed,shape):
    """Local original-puzzle domains with two alternative occupied seams."""
    rng=Random(seed)
    mate={a:b for a,b in ref-required}
    mate.update({b:a for a,b in ref-required})
    ds=list(domain_bits(p));used_cells=set();used_tiles=set();reserved=set();scope=[]
    for width in shape:
        c=next(c for c in rng.sample(range(p.n),p.n) if not {c,*map(int,p.neighbors[c,:width])}&used_cells)
        sites=[int(p.neighbors[c,s]) for s in range(width)]
        labels=[]
        for code in rng.sample(range(3*p.n),3*p.n):
            tile,rot=divmod(code,3)
            darts=[3*tile+(s-rot)%3 for s in range(width)]
            if not all(d in mate for d in darts):continue
            others=[mate[d] for d in darts]
            involved={tile,*(d//3 for d in others)}
            if len(involved)!=width+1 or involved&used_tiles:continue
            if labels and any(p.masks[code,s]==p.masks[labels[0][0],s] for s in range(width)):continue
            nbr=[3*(d//3)+(int(p.sides[c,s])-d%3)%3 for s,d in enumerate(others)]
            labels.append((code,nbr));used_tiles.update(involved)
            reserved.update(tuple(sorted((a,b))) for a,b in zip(darts,others))
            if len(labels)==2:break
        assert len(labels)==2
        ds[c]=sum(1<<code for code,_ in labels)
        for s,site in enumerate(sites):ds[site]=sum(1<<nbr[s] for _,nbr in labels)
        used_cells.update([c,*sites]);scope.append([c,*sites])
    return ds,reserved,scope


def contested(p,ref,required,seed,count=4):
    """Distinct distant cells share one uncharged tile; each paid tile implies two joins."""
    rng=Random(seed)
    mate={a:b for a,b in ref-required};mate.update({b:a for a,b in ref-required})
    ds=list(domain_bits(p));used_cells=set();used_tiles=set();reserved=set()
    for cheap in rng.sample(range(3*p.n),3*p.n):
        if all(sum(p.masks[k,0]==p.masks[cheap,s] for k in range(3*p.n))>5 for s in range(2)):break
    used_tiles.add(cheap//3)
    from validator import reverse_mask
    for j in range(count):
        c=next(c for c in rng.sample(range(p.n),p.n) if not {c,*map(int,p.neighbors[c,:2])}&used_cells)
        sites=[int(p.neighbors[c,s]) for s in range(2)]
        for paid in rng.sample(range(3*p.n),3*p.n):
            tile,rot=divmod(paid,3);darts=[3*tile+(s-rot)%3 for s in range(2)]
            if not all(d in mate for d in darts) or any(p.masks[paid,s]==p.masks[cheap,s] for s in range(2)):continue
            others=[mate[d] for d in darts]
            if len({tile,*(d//3 for d in others)})!=3 or {tile,*(d//3 for d in others)}&used_tiles:continue
            break
        else:raise AssertionError('No disjoint paid alternative')
        ds[c]=(1<<cheap)|(1<<paid);used_tiles.update([tile,*(d//3 for d in others)])
        for side,site in enumerate(sites):
            dst=int(p.sides[c,side]);m=others[side];tile2,side2=divmod(m,3)
            required_code=3*tile2+(dst-side2)%3
            target=reverse_mask(int(p.masks[cheap,side]))
            fillers=[v for v in range(3*p.n) if v//3!=cheap//3 and int(p.masks[v,dst])==target]
            ds[site]=(1<<required_code)|sum(1<<v for v in fillers)
        reserved.update(tuple(sorted((d,mate[d]))) for d in darts);used_cells.update([c,*sites])
    return ds,reserved

def choose(ref,groups,before,optional,requested,reserved,rng):
    reserve=set(reserved)
    for g in sorted(groups,key=lambda g:len(g)):
        allowed=sorted(set(g)-before)
        if allowed and not reserve.intersection(g):reserve.add(rng.choice(allowed))
    options=sorted(set(ref)-before-reserve);rng.shuffle(options)
    return set(options[:min(len(options),max(0,requested-len(before)))])

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--tiles',required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();p=SearchProblem(read_json(args.tiles));ref,limit=certified_reference(p)
    codes=read_json('docs/R29_INCUMBENT.json')['codes'];required=inventory(p).required
    _,_,groups=reference_groups(p,codes)
    rows=[]
    for mode in ('unrestricted','pinned','cover_bad_seams'):
        for seed in range(1 if mode=='unrestricted' else 8):
            for count in ([0] if mode=='unrestricted' else ([10,12,16,20,24,32] if mode=='pinned' else [0,4,12])):
                if mode=='unrestricted':ds,n=list(domain_bits(p)),p.n
                else:ds,n=screen(p,codes,seed,count,mode=='cover_bad_seams')
                before,opt=implied(p,ds,ref,required)
                for requested in (0,184,185,186):
                    public=set(before);additional=set();rng=Random(1000+seed)
                    # Keep an available departure from every known physical conflict.
                    reserve=set()
                    for g in sorted(groups,key=lambda g:len(g)):
                        allowed=sorted(set(g)-public)
                        if allowed and not reserve.intersection(g):reserve.add(rng.choice(allowed))
                    options=sorted(set(ref)-public-reserve);rng.shuffle(options)
                    additional.update(options[:min(len(options),max(0,requested-len(public)))])
                    limited={k:v for k,v in opt.items() if k not in additional};total=len(public|additional)
                    prices,each=alternative_prices(ds,limited)
                    cheap,joint,_=assignment_price(ds,prices)
                    easy=total>limit;pure=(joint is not None and total+(joint+1)//2>limit)
                    poses=sum(len(v)>limit-total>=0 for v in each.values())
                    groups_hit=sum(g<=(public|additional) for g in groups)
                    rows.append(dict(mode=mode,seed=seed,free=n,requested=requested,individual=total,prior_groups=groups_hit,
                                     independent_twice=cheap,assignment_twice=joint,over_before=easy,over_after=pure,extra_poses=poses))
    counters={mode:dict(rows=len(x),nontrivial=sum(r['assignment_twice']>0 for r in x),extra_closures=sum(r['over_after'] and not r['over_before'] for r in x),
                        beyond_cheap=sum(r['assignment_twice']>r['independent_twice'] for r in x),
                        poses=sum(r['extra_poses'] for r in x if not r['over_before']),prior_group_free_closures=sum(r['over_after'] and not r['over_before'] and not r['prior_groups'] for r in x))
              for mode in ('unrestricted','pinned','cover_bad_seams') if (x:=[r for r in rows if r['mode']==mode])}
    motif=[]
    for seed in range(4):
        for shape in ((1,),(1,1),(2,),(2,2)):
            ds,reserved,sites=alternatives(p,ref,set(required),seed,shape)
            forced,opt=implied(p,ds,ref,required)
            for requested in (185,186):
                extras=choose(ref,groups,forced,opt,requested,reserved,Random(seed+1000))
                assumed=forced|extras
                leftovers={k:v for k,v in opt.items() if k not in extras}
                cost,poses=alternative_prices(ds,leftovers)
                cheap,best,_=assignment_price(ds,cost)
                sl=limit-len(assumed)
                out=dict(seed=seed,shape=shape,sites=sites,explicit=len(extras),forced=len(assumed),slack=sl,
                         individual_over=sum(len(v)>sl for v in poses.values()),local=cheap,joint=best,
                         prior_conflicts=sum(g<=assumed for g in groups))
                if seed==0 and shape==(1,1) and requested==186:
                    from r30_gold import search
                    old=search(p,geography=True,reference_bound=True,domains=list(ds),bonds=extras,max_rounds=1,seconds=3)
                    from unittest.mock import patch
                    out['native_new']={k:old.get(k) for k in ('status','cp_calls','branches','certified_reference')}
                    with patch('r35_alternative_retention.prune_alternatives',return_value={}):
                        prior=search(p,geography=True,reference_bound=True,domains=list(ds),bonds=extras,max_rounds=1,seconds=3)
                    out['native_predecessor']={k:prior.get(k) for k in ('status','cp_calls','branches','certified_reference')}
                motif.append(out)
    hall=[]
    for seed in range(4):
        ds,reserved=contested(p,ref,set(required),seed)
        forced,opt=implied(p,ds,ref,required)
        extras=choose(ref,groups,forced,opt,185,reserved,Random(1500+seed))
        assumed=forced|extras
        cost,poses=alternative_prices(ds,{k:v for k,v in opt.items() if k not in extras});cheap,best,_=assignment_price(ds,cost)
        row=dict(seed=seed,forced=len(assumed),slack=limit-len(assumed),individual_over=sum(len(v)>limit-len(assumed) for v in poses.values()),local=cheap,joint=best,prior_conflicts=sum(g<=assumed for g in groups))
        apply=prune_alternatives(ds.copy(),{k:v for k,v in opt.items() if k not in extras},row['slack'])
        row['checked_projected']=apply
        if seed==0:
            from r30_gold import search
            from unittest.mock import patch
            for name,changed in (('native_predecessor',False),('native_new',True)):
                if changed:r=search(p,bonds=extras,domains=list(ds),reference_bound=True,geography=True,seconds=3,max_rounds=1)
                else:
                    with patch('r35_alternative_retention.prune_alternatives',return_value={}):
                        r=search(p,bonds=extras,domains=list(ds),reference_bound=True,geography=True,seconds=3,max_rounds=1)
                row[name]={k:r.get(k) for k in ('status','cp_calls','branches','certified_reference')}
        hall.append(row)
    margins=[]
    for n in (3,4):
        for seed in range(4):
            ds,reserved=contested(p,ref,set(required),seed,n);forced,opt=implied(p,ds,ref,required)
            for slack in (2,3,4,5,6):
                extras=choose(ref,groups,forced,opt,limit-slack,reserved,Random(1500+seed))
                leftovers={k:v for k,v in opt.items() if k not in extras}
                live=ds.copy();trial=prune_alternatives(live,leftovers,slack)
                second=None
                if trial['unit_poses'] and not trial['over']:
                    from r35_geographic_loss import retained_bound
                    nxt=retained_bound(p,live,ref,limit,set(required)|extras)
                    second={'forced':nxt['forced'],'over':nxt['over'],'unit_poses':nxt['unit_poses']}
                margins.append(dict(rows=n,seed=seed,slack=slack,decision=trial,second_pass=second))
    args.output.write_text(json.dumps(dict(summary=counters,rows=rows,motifs=motif,hall=hall,boundary=margins,scope=dict(puzzle_fingerprint=p.fingerprint,tiles_sha256=hashlib.sha256(Path(args.tiles).read_bytes()).hexdigest(),reference_bonds=len(ref),retention_limit=limit)),indent=2)+'\n',encoding='utf8')
    print('contested',hall)
    print(counters);print('motifs',[(r['shape'],r['slack'],r['local'],r['joint'],r['prior_conflicts']) for r in motif if r['seed']==0])


if __name__=='__main__':main()
