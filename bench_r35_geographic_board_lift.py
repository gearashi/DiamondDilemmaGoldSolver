"""Complete every unconstrained tile/rotation of a geographically packable near-reference residual."""
import argparse
from collections import Counter
from itertools import permutations, product
import json
from pathlib import Path

import numpy as np

from bench_r35_geographic_components import parts
from bench_r35_loss_boundary344 import problem_scope
from constraint_symmetry import proper_rotations
from io_utils import atomic_json
from r27_incumbent import BoardModel
from r31_geographic import physical_bonds
from r35_geographic_packing_check import component_poses


def solve(problem, reference, required, lost, output, phase, *, max_free=5):
    retained=reference-lost
    seams=retained|required
    incidence={}
    for u,a,v,b in problem.edges:
        incidence[u,a]=(v,b);incidence[v,b]=(u,a)
    components=sorted(parts(seams), key=len, reverse=True)
    poses=[list(component_poses(problem.n,incidence,c)) for c in components]
    transforms=proper_rotations(problem.board)
    def orbit(assigned):
        return {tuple(sorted((t,op.cells[cell],(rot+op.side_offsets[cell])%3)
                             for t,(cell,rot) in assigned.items())) for op in transforms}
    anchors={tuple(sorted((t,cell,rot) for t,(cell,rot) in assigned.items())):(mask,assigned)
             for mask,assigned in poses[0]}
    chosen=[]
    while anchors:
        key=min(anchors)
        copies=orbit(anchors[key][1])
        if not copies<=anchors.keys() or len(copies)!=len(transforms):
            raise ArithmeticError('Anchor pose list disagrees with the exact proper rotation orbit.')
        chosen.append(anchors[key])
        for copy in copies:del anchors[copy]
    poses[0]=chosen
    model=BoardModel(problem)
    fixed_tiles={dart//3 for e in seams for dart in e}
    free=sorted(set(range(problem.n))-fixed_tiles)
    if len(free)>max_free:
        raise ValueError('This finite completion control is limited to small free tile sets.')
    fillings=np.array([[3*t+r for t,r in zip(order,turns)]
                       for order in permutations(free) for turns in product(range(3),repeat=len(free))],dtype=np.int16)
    hist=Counter()
    peak=-1;best=None;partial=0
    board=np.full(problem.n,-1,dtype=np.int16)
    for combo in product(*poses):
        occupied=0
        for mask,_ in combo:
            if occupied&mask:break
            occupied|=mask
        else:
            partial+=1;board.fill(-1)
            for _,assigned in combo:
                for t,(cell,rot) in assigned.items():board[cell]=3*t+rot
            empty=np.flatnonzero(board<0)
            if len(empty)!=len(free):raise ArithmeticError('Rigid poses used a cell or tile more than once.')
            near=np.isin(model.a,empty)|np.isin(model.b,empty)
            base=int(model.matches(board)[~near].sum())
            rows=np.tile(board,(len(fillings),1));rows[:,empty]=fillings
            match=model.masks[rows[:,model.a[near]],model.sa[near]]==model.reverse[model.masks[rows[:,model.b[near]],model.sb[near]]]
            scores=base+match.sum(axis=1);values,counts=np.unique(scores,return_counts=True)
            hist.update(dict(zip(map(int,values),map(int,counts))))
            idx=int(scores.argmax())
            if int(scores[idx])>peak:
                peak=int(scores[idx]);best=rows[idx].tolist()
                checked=problem.validate(best)
                if checked['matched_edges']!=peak or not checked['unique_tiles'] or not seams<=set(physical_bonds(problem,best)):
                    raise ArithmeticError('Independent full-board checker disagrees with the retained-seam completion.')
                if peak>207:
                    atomic_json(output.with_name(output.stem+'-improvement.json'),dict(fingerprint=problem.fingerprint,
                                score=peak,codes=best,validation=checked,phase=hex(phase)))
            if int(scores.max())==len(problem.edges):
                for i in np.flatnonzero(scores==len(problem.edges)):
                    codes=rows[i].tolist();validation=problem.validate(codes)
                    if validation['valid']:
                        atomic_json(output,dict(Gold=True,phase=hex(phase),codes=codes,validation=validation))
                        return
    if best is None:raise ArithmeticError('A verified residual had no physical completions.')
    checked=problem.validate(best)
    result=dict(fingerprint=problem.fingerprint,phase=hex(phase),retained=len(retained),required=len(required),
                components=[len({dart//3 for pair in c for dart in pair}) for c in components],free_tiles=free,
                pose_counts_after_anchor=[len(r) for r in poses],proper_rotations=len(transforms),
                partial_placements=partial,completions_per_placement=len(fillings),rotation_orbits=sum(hist.values()),
                labelled_boards=len(transforms)*sum(hist.values()),score_histogram=dict(sorted(hist.items())),
                best_score=peak,codes=best,validation=checked)
    atomic_json(output,result)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--tiles',required=True);cli.add_argument('--receipt',required=True,type=Path)
    cli.add_argument('--phase',required=True,type=lambda v:int(v,0));cli.add_argument('--output',required=True,type=Path)
    a=cli.parse_args()
    p,ref,_,req=problem_scope(a.tiles);d=json.loads(a.receipt.read_bytes())
    if d['puzzle']!=p.fingerprint:raise ValueError('Original physical dataset required.')
    if 'boundaries' in d and int(d['phase'],16)==a.phase:
        row=next(v for v in d['boundaries'] if v['status']=='pose packing feasible')
        lost=row['lost']
    else:
        row=next(x for x in d['separations'] if int(x['phase'],16)==a.phase and x['status']=='pose packing feasible')
        lost=row['witness_departures']
    solve(p,ref,req,set(map(tuple,lost)),a.output,a.phase)
