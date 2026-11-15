"""Replay the Round 34 physical follow-up controls. Use --root-lifts for bounded Gold lifts."""
import argparse
from collections import Counter
import json

import numpy as np

from constraint_models import solve
from r27_incumbent import BoardModel
from r27_structure_filters import domain_bits
from r27_structure_fixtures import planted
from r28_exchange import read_json
from r29_gold_phase import SignedParity
from r31_sampling import ranked_start
from r32_fresh_geometry import skeleton_domains
from r32_fresh_phases import independent_phases, phase_cube, SignedRepair, lift_cube
from r32_fresh_skeleton import parity_seed
from r32_gold_bonds import inventory
from r34_contour_walk import contour_walk
from r34_geographic_mates import _post
from search_problem import SearchProblem
from test_r34_physical_followups import ring_puzzle


def phase_results(p):
    order=independent_phases(p)
    contraction={}
    for name,seq in (('numbered',range(1,p.n)),('independent',order)):
        q=SignedParity(p);q.pin(0,0)
        for a,b in sorted(inventory(p).required):q.bond(a,b)
        rank=len(q.base.pivots);counts={}
        for i,t in enumerate(seq):
            mark=len(q.base.trail)
            if not q.pin(t,0):
                q.base.rollback(mark)
                assert q.pin(t,1)
            if i+1 in (2,4,8,16,32,40):
                counts[i+1]=[len(q.base.pivots)-rank,len(q.base.forced())]
            if i+1==40:break
        contraction[name]=counts
    finite=ring_puzzle()
    fixed=[1<<(3*i) for i in range(finite.n)]
    decisions=[]
    for independent,width in ((False,finite.n-1),(True,len(independent_phases(finite)))):
        status=Counter()
        for i in range(1<<width):
            pins=phase_cube(finite,width,i,independent=independent)
            row=lift_cube(finite,inventory(finite).required,pins,fixed,2)
            status[row['status']]+=1
        decisions.append(dict(status))
    return dict(dimension=len(order),contraction=contraction,finite_lifts=decisions)


def mate_results(p):
    data={}
    for split in (False,True):
        q,ds=planted(1,5,34,split,True)
        q.domains=[[v for v in range(3*q.n) if d>>v&1] for d in ds]
        for engine in ('sat','cp-sat','cp-sat-circuit','cp'):
            groups=[]
            for enabled in (False,True):
                q.physical_gold_support=enabled
                row=solve(q,engine=engine,target='gold',seconds=10)
                groups.append(dict(status=row['status'],nodes=row['nodes']))
            data[str(split)+'_'+engine]=groups
    return dict(restrictions=_post(p,p.domains,lambda *args:None,lambda:None),controls=data)


def donor_results(p):
    m=BoardModel(p);ban=inventory(p).forbidden
    tally=Counter()
    for seed in range(8):
        rng=np.random.default_rng(seed)
        phases=rng.integers(0,2,p.n).tolist()
        b=SignedRepair(m,phases,ban)
        board=ranked_start(m,seed)
        for _ in range(16):
            cells=m.independent_set(board,rng)
            board=b.block(board,cells)
        tally.update(b.stats)
    return dict(counts=dict(tally),raw_vector_builds=3*tally['rows_computed'],
                template_bytes=sum(row.nbytes for row in b.geometry.templates.values()),
                final_phase_bytes=sum(row.nbytes for row in b.prices.values()))


def contour_results(p, full):
    def controls(q,limit,lifts):
        independent=[parity_seed(q,s,conditioned=True) for s in range(limit)]
        walk=contour_walk(q,0,limit=limit)
        sets=[[r['patterns'] for r in independent],walk['patterns']]
        data=dict(independent_cells=limit*q.n,walk_cells=walk['assignment_cells'],
                  unique=[len({tuple(row) for row in group}) for group in sets],
                  score=[max(r['actual_matches'] for r in independent),walk['best']['matched_edges']])
        if lifts:
            root=domain_bits(q)
            data['lifts']=[dict(Counter(lift_cube(q,inventory(q).required,{0:0},
                skeleton_domains(q,root,pat),2)['status'] for pat in group[:lifts])) for group in sets]
        return data
    return dict(root=controls(p,16,3 if full else 0),ring=controls(ring_puzzle(),6,6))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiles',default='data/tiles.json')
    parser.add_argument('--root-lifts',action='store_true')
    args=parser.parse_args()
    puzzle=SearchProblem(read_json(args.tiles))
    print(json.dumps(dict(phases=phase_results(puzzle),mates=mate_results(puzzle),donors=donor_results(puzzle),
                          contours=contour_results(puzzle,args.root_lifts)),indent=2))


if __name__=='__main__':
    main()
