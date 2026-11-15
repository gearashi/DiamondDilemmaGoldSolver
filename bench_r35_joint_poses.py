"""Check the complete labelled component projection and published contour-to-region controls."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from random import Random

from constraint_advanced import _complete_matching
from constraint_pruning import Pruner,_tile_choices
from r27_structure_filters import bits,domain_bits
from r27_structure_fixtures import independent,planted
from r28_exchange import read_json
from r28_geo_sewing import PosePacking
from r30_gold import search
from r32_fresh_geometry import skeleton_domains
from r32_fresh_skeleton import parity_seed
from r32_gold_bonds import inventory
from r34_contour_walk import contour_walk
from r35_joint_poses import narrowed_poses,branch_domains
from search_problem import SearchProblem
from test_r34_physical_followups import ring_puzzle
from validator import reverse_mask


def prior(rules,ds):
    ds=list(ds)
    while all(ds) and rules.prune_clusters(ds):pass
    return tuple(ds)


def adverse(p,rules,seed):
    rng=Random(seed)
    group0,group1=rules.clusters[:2]
    options=[list(group[2]) for group in (group0,group1)]
    for group in options:rng.shuffle(group)
    def cells(pose):return {c for c,v in pose}
    a=options[0][0];i,j=sorted(cells(a))
    b=next(pose for pose in options[1] if cells(pose)&cells(a)=={i})
    c=next(pose for pose in options[1] if cells(pose)&cells(a)=={j} and not cells(pose)&cells(b))
    a0=next(pose for pose in options[0] if not cells(pose)&(cells(a)|cells(b)|cells(c)))
    ds=list(domain_bits(p))
    for group,keep in ((group0,(a,a0)),(group1,(b,c))):
        for k in range(p.n):ds[k]&=~group[1]
        for pose in keep:
            for place,code in pose:ds[place]|=1<<code
    before=prior(rules,ds)
    state=narrowed_poses(before,rules.clusters,work_limit=2000000)
    assert all(state['domains']) and all(before[k]>>v&1 for k,v in a)
    assert all(not (state['domains'][k]>>v&1) for k,v in a)
    children=branch_domains(state)
    return dict(seed=seed,failure_pose=a,supported_alternative=a0,conflicting_choices=[b,c],
                prior_removed=sum((x^y).bit_count() for x,y in zip(ds,before)),joint=state['stats'],
                complete_projection=state['complete'],distinct_children=len(set(children)))


def finite_region(p,ds):
    match=_complete_matching([_tile_choices(d) for d in ds])
    if match is None:return dict(status='INFEASIBLE',reason='no tile matching')
    board=[next(v for v in bits(ds[c]) if v//3==t) for c,t in enumerate(match)]
    movable=range(min(4,p.n))
    pool=sum(7<<(3*match[c]) for c in movable)
    small=tuple(d&pool if c in movable else 1<<board[c] for c,d in enumerate(ds))
    for a,s,b,t in p.edges:
        if a not in movable and b not in movable and int(p.masks[board[a],s])!=reverse_mask(int(p.masks[board[b],t])):
            return dict(status='INFEASIBLE',reason='fixed incompatible seam',seam=[a,s,b,t],
                        codes=[board[a],board[b]])
    answers=independent(p,small)
    return dict(status='GOLD' if answers else 'INFEASIBLE',exact_solutions=len(answers),
                witness=list(min(answers)) if answers else None)


def contours(p,n):
    rules=Pruner(p,regions=[],parity=False,gold_bonds=True)
    ds=domain_bits(p)
    direct=[parity_seed(p,seed,conditioned=True)['patterns'] for seed in range(n)]
    walk=contour_walk(p,0,limit=n)['patterns']
    observed={}
    report={}
    for mode,patterns in [('independent',direct),('two_flip_walk',walk),('repeated_first',direct[:1]*3)]:
        trials=[]
        for raw in patterns:
            pat=tuple(raw)
            if pat not in observed:
                d=skeleton_domains(p,ds,pat);baseline=prior(rules,d)
                key=hashlib.sha256(bytes(pat)).hexdigest()
                if not all(baseline):
                    observed[pat]=dict(hash=key,status='NO_INDEPENDENT_MANDATORY_POSE')
                else:
                    joined=narrowed_poses(baseline,rules.clusters,work_limit=250000)
                    if not all(joined['domains']):
                        observed[pat]=dict(hash=key,status='NO_JOINT_SUPPORT',work=joined['stats'])
                    else:
                        observed[pat]=dict(hash=key,status='JOINT_SURVIVOR' if joined['stats'].get('support_witnesses') else 'JOINT_UNDECIDED',scope_complete=joined['complete'],
                                           removed=joined['stats']['removed_codes'],region=finite_region(p,joined['domains']))
            trials.append(observed[pat]['hash'])
        report[mode]=trials
    return dict(runs=report,unique=list(observed.values()))


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--tiles',default='data/tiles.json')
    cli.add_argument('--output',type=Path)
    args=cli.parse_args();p=SearchProblem(read_json(args.tiles))
    rules=Pruner(p,regions=[],parity=False,gold_bonds=True)
    base=domain_bits(p)
    root=narrowed_poses(base,rules.clusters,work_limit=2000000)
    out=dict(root=dict(complete=root['complete'],**root['stats']),
             restricted_real=[adverse(p,rules,seed) for seed in (35,71,103,157)],
             original_contours=contours(p,16),held_ring_contours=contours(ring_puzzle(),6))
    out['planted']=[]
    for seed in (35,71,103,157):
        q,d=planted(1,4,seed)
        r=Pruner(q,regions=[],parity=False,gold_bonds=True)
        old=prior(r,d)
        state=narrowed_poses(old,r.clusters,work_limit=1000000)
        real=independent(q,d)
        assert real and independent(q,state['domains'])==real
        kids=branch_domains(state)
        archived=[independent(q,child) for child in kids]
        assert not kids or (set.union(*archived)==real and sum(map(len,archived))==len(real))
        def run(ds):
            rec=search(q,domains=ds,geography=True,channel=True,seconds=5)
            return {k:rec[k] for k in ('status','cp_calls','branches')}
        out['planted'].append(dict(seed=seed,baseline=run(old),joint=run(state['domains']),valid=len(real),
                                   work=state['stats'],branches=len(kids)))
    text=json.dumps(out,indent=2)+'\n'
    if args.output:args.output.write_text(text,encoding='utf-8')
    else:print(text)


if __name__=='__main__':main()
