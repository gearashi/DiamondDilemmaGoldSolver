"""Price incidental seams in fresh speculative nine-tile components and replay immutable mate failures."""
import argparse,json
from collections import Counter
from pathlib import Path
from random import Random

from r27_incumbent import BoardModel
from r28_exchange import read_json
from r31_fresh import Assemblies
from r32_gold_bonds import inventory
from r35_pose_contacts import Boundaries
from r35_speculative_poses import PoseJoins
from search_problem import SearchProblem


def grow(problem,model,assembly,seed):
    rng=Random(seed);joins=PoseJoins(assembly);groups=list(joins.base);start=rng.randrange(len(groups))
    group=groups[start];required=assembly.bonds;used={s for e in required for s in e}
    possible=list(sorted(inventory(problem).possible));rng.shuffle(possible)
    edges=[]
    for _ in range(12):
        for edge in possible:
            a,b=edge
            if a in used or b in used or ((a//3 in group[0])==(b//3 in group[0])):continue
            proposed=joins.append(groups,edge)
            if proposed is None:continue
            groups=proposed;group=groups[-1];used.update(edge);edges.append(edge);break
        else:break
        if len(group[0])==9:break
    if len(group[0])!=9:return None
    return edges,group


def main():
    cli=argparse.ArgumentParser();cli.add_argument('--tiles',type=Path,required=True);cli.add_argument('--output',type=Path,required=True);cli.add_argument('--count',type=int,default=32)
    args=cli.parse_args();p=SearchProblem(read_json(args.tiles));m=BoardModel(p);a=Assemblies(m)
    physical=Boundaries(p,True);ordinary=Boundaries(p,False)
    res=dict(puzzle=p.fingerprint,records=[])
    for seed in range(8500,8500+args.count):
        grown=grow(p,m,a,seed)
        if grown is None:continue
        edges,group=grown;rows=group[1]
        counts=Counter();raw=Counter();native_forbidden=Counter();exact_contact=Counter()
        for pose in rows:
            _,_,_,raw_bad=ordinary.describe(pose)
            _,_,_,physical_bad=physical.describe(pose)
            raw[raw_bad]+=1;counts[physical_bad]+=1;native_forbidden[physical_bad-raw_bad]+=1
            assignments=dict(pose);touches=0
            for c,v in pose:touches+=sum(b in assignments and c<b for b,j in p.board.neighbors[c])
            exact_contact[touches]+=1
        out=dict(seed=seed,edges=edges,members=sorted(group[0]),labelled=len(rows),raw_incidental=dict(raw),gold_incidental=dict(counts),
                 frozen_forbidden=dict(native_forbidden),interior_contacts=dict(exact_contact))
        res['records'].append(out)
        args.output.write_text(json.dumps(res,indent=2)+'\n',encoding='utf8')
    print(len(res['records']),sum(r['labelled'] for r in res['records']),flush=True)


if __name__=='__main__':main()
