"""Compare geographic pose contact filtering against the same footprint-only component packings."""
import argparse,json
from collections import Counter,defaultdict
from pathlib import Path
from random import Random
from unittest.mock import patch

from bench_r35_joint_poses import adverse,prior
from constraint_pruning import Pruner
from r27_structure_filters import domain_bits,Work
from r28_exchange import read_json
from r35_joint_poses import narrowed_poses,branch_domains
from r35_pose_contacts import compatible_poses
from search_problem import SearchProblem


def families(ds,clusters):
    out=[]
    for _,mask,poses in clusters:
        required=sum(1<<c for c,row in enumerate(ds) if not row&~mask);table=defaultdict(list)
        for pose in poses:
            foot=sum(1<<c for c,v in pose)
            if not required&~foot and all(ds[c]>>v&1 for c,v in pose):
                table[foot].append(pose)
        out.append(table)
    return out


def root_contact(p,ds,cs):
    work=Work(500000);groups,metrics,done=compatible_poses(p,families(ds,cs),work,restricted=False)
    return dict(metrics=metrics,cost=work.counters,done=done,labelled=sum(len(r) for g in groups for r in g.values()))


def project(p,ds,cs,opts):
    state=narrowed_poses(ds,cs,work_limit=2000000,problem=None if opts is None else p,contact_options=opts)
    return dict(complete=state['complete'],work=state['stats'],codes=sum(v.bit_count() for v in state['domains']),
                children=len(branch_domains(state)),possible=all(state['domains']))


def conditional_banned(p):
    from r32_gold_bonds import inventory,endpoints,loop_groups
    from r35_pose_contacts import Boundaries
    from test_r35_joint_poses import cluster
    inv=inventory(p);allowed=defaultdict(set)
    for a,b in inv.possible:allowed[a].add(b);allowed[b].add(a)
    examples={}
    sides=[[i+1 for i in range(11) if int(p.masks[3*(d//3),d%3])>>i&1] for d in range(3*p.n)]
    arcs=endpoints(p)
    for a,b in sorted(inv.forbidden):
        first=a//3;second=b//3
        def find():
            for root in range(min(4,p.n)):
                for rot in range(3):
                    neighbor,s=p.board.neighbors[root][(rot+a%3)%3]
                    q=(s-b%3)%3
                    for d in range(3*first,3*first+3):
                        if d==a:continue
                        l,sl=p.board.neighbors[root][(rot+d%3)%3]
                        for e in range(3*second,3*second+3):
                            if e==b:continue
                            r,sr=p.board.neighbors[neighbor][(q+e%3)%3]
                            if len({root,neighbor,l,r})!=4:continue
                            for donor in sorted(allowed[d]):
                                for other in sorted(allowed[e]):
                                    if len({first,second,donor//3,other//3})!=4:continue
                                    p1=((root,3*first+rot),(l,3*(donor//3)+(sl-donor%3)%3))
                                    p2=((neighbor,3*second+q),(r,3*(other//3)+(sr-other%3)%3))
                                    if raw.describe(p1+p2)[3]:continue
                                    return p1,p2
        raw=Boundaries(p,False);found=find()
        if found is None:continue
        p1,p2=found
        groups=[cluster([v//3 for _,v in pose],[pose]) for pose in (p1,p2)]
        ds=domain_bits(p)
        raw_state=narrowed_poses(ds,groups,problem=p,contact_options={'physical':False})
        physical=narrowed_poses(ds,groups,problem=p)
        assert all(raw_state['domains']) and not any(physical['domains'])
        kind='sealed_loop' if loop_groups(arcs,sides,[(a,b)],p.total_segments) else 'matching_only'
        examples.setdefault(kind,[]).append(dict(bond=[a,b],poses=[p1,p2]))
    return examples


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--tiles',type=Path,required=True);cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args();p=SearchProblem(read_json(args.tiles));ds=domain_bits(p)
    rules=Pruner(p,regions=[],parity=False,gold_bonds=True);cs=rules.clusters
    results=dict(input=p.fingerprint,root=root_contact(p,ds,cs),cases=[],conditional_banned=conditional_banned(p))
    for seed in (35,71,103,157):
        capture=[]
        def save(d,g,**kw):
            capture.append(d);return narrowed_poses(d,g,**kw)
        with patch('bench_r35_joint_poses.narrowed_poses',save):adverse(p,rules,seed)
        current=capture[0];runs={}
        for name,opts in [('footprint',None),('edge_masks',dict(physical=False)),('physical_pairs',dict(joint=False)),('all_queries',dict(cover_test=False)),('joint',{})]:
            runs[name]=project(p,current,cs,opts)
        results['cases'].append(dict(kind='released_restricted',seed=seed,runs=runs))
        args.output.write_text(json.dumps(results,indent=2)+'\n',encoding='utf8')
        print(seed,[(k,v['codes']) for k,v in runs.items()],flush=True)
    for seed in (85,850,851,852,853,854):
        rng=Random(seed);current=list(ds);center=rng.randrange(p.n);patch_cells={center}
        for _ in range(3):patch_cells.update(b for c in list(patch_cells) for b,_ in p.board.neighbors[c] if b>=0)
        for members,mask,embeddings in cs[:4]:
            local=[pose for pose in embeddings if all(c in patch_cells for c,v in pose)]
            chosen=rng.sample(local,4)
            for c in range(p.n):current[c]&=~mask
            for pose in chosen:
                for c,v in pose:current[c]|=1<<v
        current=prior(rules,current);runs={}
        for name,opts in [('footprint',None),('edge_masks',dict(physical=False)),('physical_pairs',dict(joint=False)),('all_queries',dict(cover_test=False)),('joint',{})]:
            runs[name]=project(p,current,cs,opts)
        results['cases'].append(dict(kind='four_local_dominoes',seed=seed,runs=runs))
        args.output.write_text(json.dumps(results,indent=2)+'\n',encoding='utf8')
        print(seed,[(k,v['codes']) for k,v in runs.items()],flush=True)

if __name__=='__main__':main()
