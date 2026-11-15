"""Check fixed, partial and unfixed tile signs on the exact same incoming pose families."""
import argparse,json
from pathlib import Path
from random import Random
from unittest.mock import patch

from bench_r35_incidental_joins import grow
from bench_r35_joint_poses import adverse
from constraint_pruning import Pruner
from r27_incumbent import BoardModel
from r28_exchange import read_json
from r29_gold_phase import EndpointCuts
from r31_fresh import Assemblies
from r32_gold_bonds import inventory
from r35_joint_poses import narrowed_poses
from r35_pose_contacts import Boundaries,PhaseRelations,PoseIndex
from search_problem import SearchProblem


def pair_cycles(p,groups,raw,relations):
    arrays=[[raw.describe(row) for row in poses] for _,poses in groups]
    checked=0;negatives=0
    for j,group in enumerate(groups):
        idx=PoseIndex(arrays[j],raw);full=(1<<len(arrays[j]))-1
        for i in range(j):
            for sig in arrays[i]:
                seen=[0,0];allowed=idx.allowed(sig,full)
                for cell,side,a in sig[1]:
                    for b,mask in idx.labels.get((cell,side),{}).items():
                        if not raw.accepts(a,b):continue
                        _,x=relations.parts[a//3];_,y=relations.parts[b//3]
                        seen[x^y^relations.required(a,b)]|=mask
                negatives+=(seen[0]&seen[1]&allowed).bit_count();checked+=len(arrays[j])
    return dict(checked=checked,relative_sign_exclusions=negatives)


def main():
    cli=argparse.ArgumentParser();cli.add_argument('--tiles',type=Path,required=True);cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args();p=SearchProblem(read_json(args.tiles));m=BoardModel(p);a=Assemblies(m)
    rules=Pruner(p,regions=[],parity=False,gold_bonds=True);raw=Boundaries(p)
    phase=int(read_json(Path(__file__).parent/'docs/R35_MIXED_MIP_DONOR.json')['phase'],0)
    inv=inventory(p)
    relation=PhaseRelations(p,inv.required)
    independent=EndpointCuts(p).sides
    assert all(relation.required(i,j)==independent[i][1]^independent[j][1]^(independent[i][0].bit_count()%2) for i,j in inv.possible)
    result=dict(relative_pairs=pair_cycles(p,a.groups,raw,relation),restricted=[],nine=[])
    for seed in (35,71,103,157):
        d=[]
        def capture(ds,group,**kw):d.append(ds);return narrowed_poses(ds,group,**kw)
        with patch('bench_r35_joint_poses.narrowed_poses',capture):adverse(p,rules,seed)
        cases={}
        for name,stride in [('free',0),('quarter',4),('half',2),('full',1)]:
            pins={i:(phase>>i)&1 for i in range(0,p.n,stride)} if stride else {}
            q=narrowed_poses(d[0],rules.clusters,problem=p,pins=pins,work_limit=2000000)
            cases[name]=dict(codes=sum(row.bit_count() for row in q['domains']),labels=sum(map(len,q['poses'])),stats=q['stats'])
        result['restricted'].append(dict(seed=seed,cases=cases))
    for seed in range(8500,8564):
        edges,g=grow(p,m,a,seed)
        free=PhaseRelations(p,[*inv.required,*edges]);assert free.valid
        signed=Boundaries(p,phases=free)
        raw_count=sum(raw.describe(row)[3]==0 for row in g[1])
        signed_count=sum(signed.describe(row)[3]==0 for row in g[1])
        result['nine'].append(dict(seed=seed,raw=raw_count,relative=signed_count))
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(result['relative_pairs'],[(r['seed'],[q['codes'] for q in r['cases'].values()]) for r in result['restricted']],
          sum(r['raw']-r['relative'] for r in result['nine']),flush=True)
if __name__=='__main__':main()
