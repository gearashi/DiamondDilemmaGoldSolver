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

def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--tiles',type=Path,required=True);cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args();p=SearchProblem(read_json(args.tiles));ds=domain_bits(p)
    rules=Pruner(p,regions=[],parity=False,gold_bonds=True);cs=rules.clusters
    results=dict(input=p.fingerprint,root=root_contact(p,ds,cs),cases=[])
    for seed in (35,71,103,157):
        capture=[]
        def save(d,g,**kw):
            capture.append(d);return narrowed_poses(d,g,**kw)
        with patch('bench_r35_joint_poses.narrowed_poses',save):adverse(p,rules,seed)
        current=capture[0];runs={}
        for name,opts in [('footprint',None),('edge_masks',dict(physical=False)),('physical_pairs',dict(joint=False)),('joint',{})]:
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
        for name,opts in [('footprint',None),('edge_masks',dict(physical=False)),('physical_pairs',dict(joint=False)),('joint',{})]:
            runs[name]=project(p,current,cs,opts)
        results['cases'].append(dict(kind='four_local_dominoes',seed=seed,runs=runs))
        args.output.write_text(json.dumps(results,indent=2)+'\n',encoding='utf8')
        print(seed,[(k,v['codes']) for k,v in runs.items()],flush=True)

if __name__=='__main__':main()
