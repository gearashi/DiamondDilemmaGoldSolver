"""Measure Gold loss bounds, proof reuse and production support reductions for round 32."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
import json
from itertools import combinations
from time import monotonic

from constraint_dfs import _build_support_tables
from constraint_pruning import Pruner
from r27_structure_fixtures import planted
from r27_structure_filters import domain_bits
from r30_gold import GoldMatching
from r31_geographic import physical_bonds
from r32_gold_bonds import inventory, bounded_loop_cores
from r32_gold_loss import (read_json, reference_groups, pose_lower, master, dropout, geographic,
                           first_loss_partition, Certificates)
from r32_gold_partition import maximize_disjoint, ParityLoss
from r32_gold_syndrome import OutsideSyndromes
from r32_gold_watch import placed_bonds, verified_placement, BondWatcher
from search_problem import SearchProblem


def timed(fn,*args,**kwargs):
    start=monotonic()
    result=fn(*args,**kwargs)
    result['wall']=round(monotonic()-start,3)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiles',default='data/tiles.json')
    parser.add_argument('--center',default='docs/R29_INCUMBENT.json')
    parser.add_argument('--output',required=True)
    parser.add_argument('--max-budget',default=15,type=int)
    parser.add_argument('--layer-seconds',default=120,type=int)
    args=parser.parse_args()
    p=SearchProblem(read_json(args.tiles),physical_gold_support=True)
    board=read_json(args.center)['codes']
    ref,cert,groups=reference_groups(p,board)
    inv=inventory(p)
    ds=domain_bits(p)
    q=OutsideSyndromes(p,ref,groups,inv.required)
    group=q.groups
    out={'binding':p.fingerprint,'matched':len(ref),'possible':len(inv.possible),'required':sorted(inv.required),
         'forbidden':sorted(inv.forbidden),'partial_loop_bonds':sorted(inv.loops),
         'disjoint_groups':[sorted(g) for g in group],
         'input_claims':cert.clauses(),'outside_all':len(ref-set().union(*group)),'outside_variable':len(q.outside)}
    out['baseline_required']=len(Pruner(p,regions=[],parity=False).bonds)
    a=_build_support_tables(p,ds,'edges',lambda:None)
    b=_build_support_tables(p,ds,'gold',lambda:None)
    out['directed_pose_support_words_removed']=sum((x^y).bit_count() for aa,bb in zip(a.compatible,b.compatible)
                                                 for ra,rb in zip(aa,bb) for x,y in zip(ra,rb))
    out['matching_vars']=[len(GoldMatching(p,ds,None,[],physical_support=k).mate) for k in (False,True)]
    out['mask_baseline']=timed(dropout,p,ref,12,(),2,'xor')
    rows=[]
    for mode in ('xor','integer','euler','mating'):
        v=timed(dropout,p,ref,16,groups,2,mode,inv.required,inv.forbidden)
        rows.append({k:v for k,v in v.items() if k not in ('phases','losses')})
    out['relaxations']=rows
    poses=pose_lower(p,ref,groups,board,4)
    out['poses']=poses
    out['pinned_pose']=pose_lower(p,ref,groups,board,2,[0])
    master_answer=timed(master,p,ref,cert,16,3,inv.required,inv.forbidden)
    out['master']={k:v for k,v in master_answer.items() if k!='clauses'}
    out['master']['checked_claims_after']=len(master_answer['clauses'])
    author=Certificates(p)
    author.add(master_answer['clauses'])
    examples=placed_bonds(p,enumerate(board))
    units=[]
    for g in group:
        vv=sorted({pose for side in g for pose in examples[side]})
        for c,code in vv:
            masks=list(ds)
            for pos,v in vv:
                if c!=pos:masks[pos]=1<<v
            narrowed=list(masks)
            watcher=BondWatcher(p,[g]);watcher.prune(narrowed)
            if watcher.removed:
                units.append(dict(group=len(g),placements=len(vv),removed=watcher.removed,
                                  rechecked=verified_placement(p,g,vv)))
                break
    out['watched_units']=units
    extras,visits=bounded_loop_cores(p,inv.required,40000)
    out['local_loops']={'queries':visits,'examples':sorted([sorted(c) for c in extras],key=lambda c:(len(c),c))[:20],
                        'distinct':len(extras),'sizes':sorted({len(c) for c in extras})}
    restricted,rd=planted(1,2,23,False,True)
    known=[3*i for i in range(restricted.n)]
    pr,cc,small=reference_groups(restricted,known)
    with TemporaryDirectory() as tmp:
        db=str(Path(tmp)/'geo.sqlite')
        found=geographic(restricted,pr,small,cc.rows,[],0,0,None,4,db,rd)
        replay=geographic(restricted,pr,small,cc.rows,[],0,0,None,4,db,rd)
        out['geographic_reopen']=[found['status'],bool(found.get('codes')),replay['cp_calls_now']]
    out['root_branches']=len(first_loss_partition(group)[2])
    # Reproduce the prior 12-loss and new 14/15 global results. A single query
    # at upper K already covers all budgets <=K. Direct comparison uses the
    # independent selected-bond deletion traversal on the complete 12 loss layer.
    bound=ParityLoss(p,ref,groups,inv.required)
    out['direct_twelve']=[timed(bound.interval,12,e,0,None,30) for e in range(3)]
    out['unindexed_fourteen']=[timed(q.check,14,e,args.layer_seconds) for e in (4,)]
    out['projected_layers']=[]
    for k in range(14,args.max_budget+1):
        for e in range(k-len(group)+1):
            row=timed(q.indexed,k,e,args.layer_seconds)
            out['projected_layers'].append(row)
            print(k,e,row['status'],row['wall'],flush=True)
            if row['status']!='INFEASIBLE':break
    Path(args.output).write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
