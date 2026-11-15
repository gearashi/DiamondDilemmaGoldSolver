"""Compare seed construction on independent original starts and conditional physical domains."""
import argparse
from collections import Counter
from copy import copy
from pathlib import Path

import numpy as np

from io_utils import atomic_json
from r27_incumbent import BoardModel
from r28_exchange import read_json
from r30_fresh import fresh_start
from r31_fresh import Assemblies, physical_bonds
from search_problem import SearchProblem


def conditional(source,hint,witness,rng,extras):
    model=copy(source.model)
    model.legal=np.zeros_like(model.legal)
    cells=np.arange(model.n)
    model.legal[cells,hint]=True
    model.legal[cells,witness]=True
    # Add individually legal component poses which can compete for scarce
    # ordinary slots. The complete independent witness remains in every domain.
    for _,rows in source.groups:
        for i in rng.choice(len(rows),extras,replace=False):
            for c,v in rows[i]:
                model.legal[c,v]=True
    result=copy(source)
    result.model=model
    result.groups=[(tiles,[row for row in poses if all(model.legal[c,v] for c,v in row)])
                   for tiles,poses in source.groups]
    result.groups.sort(key=lambda item:len(item[1]))
    return result


def measured(p,obj,hint,seed,work,modes):
    policies={}
    for name in modes:
        mode=None if name=='released' else name
        stats=Counter()
        try:
            result=obj.seed(hint,np.random.default_rng(seed),work=work,constructor=mode,stats=stats)
        except ValueError as error:
            if str(error)!='No legal mandatory-bond start found within the pose budget.':raise
            score=None
        else:
            obj.model.legal_board(result)
            bonds=physical_bonds(obj.model,result)
            if not obj.bonds<=bonds or obj.forbidden&bonds:raise ArithmeticError('Bad physical mate.')
            score=p.validate(result.tolist())['matched_edges']
        policies[mode or 'released']=dict(score=score,work=dict(stats))
    return policies


def contours(p,output,work,modes):
    import hashlib
    from r27_structure_filters import domain_bits
    from r32_fresh_geometry import skeleton_domains
    from r32_fresh_skeleton import parity_seed
    from r34_contour_walk import contour_walk
    from r35_pose_constructor import bits
    model=BoardModel(p)
    a=Assemblies(model)
    ds=domain_bits(p)
    direct=[parity_seed(p,seed,conditioned=True)['patterns'] for seed in range(16)]
    walk=contour_walk(p,0,limit=16)['patterns']
    raw=direct+walk+direct[:1]*3
    seen={}
    full=read_json(Path(__file__).parent/'docs/R35_JOINT_POSES.json')['original_contours']['unique']
    for index,pattern in enumerate(raw):
        fingerprint=hashlib.sha256(bytes(pattern)).hexdigest()
        physical=tuple(skeleton_domains(p,ds,pattern))
        if physical in seen:
            if seen[physical]['hash']!=fingerprint:raise AssertionError('Unexpected different source pattern.')
            continue
        m=copy(model)
        m.legal=np.zeros_like(model.legal)
        for c,mask in enumerate(physical):
            m.legal[c,list(bits(mask))]=True
        aa=copy(a)
        aa.model=m
        aa.groups=sorted(((tiles,[row for row in poses if all(m.legal[c,v] for c,v in row)]) for tiles,poses in a.groups),
                         key=lambda group:len(group[1]))
        seed=353900+index
        hint=fresh_start(m,np.random.default_rng(seed))
        seen[physical]=dict(hash=fingerprint,seed=seed,policies=measured(p,aa,hint,seed,work,modes))
        atomic_json(output,dict(input=p.fingerprint,budget=work,emitted=len(raw),records=list(seen.values())))
        print(fingerprint[:8],[(k,v['score']) for k,v in seen[physical]['policies'].items()],flush=True)
    assert set(item['hash'] for item in seen.values())==set(item['hash'] for item in full)


def repeated_starts(p,output,seeds):
    a=Assemblies(BoardModel(p))
    rows=[]
    for seed in seeds:
        hint=fresh_start(a.model,np.random.default_rng(seed))
        cold=copy(a)
        cold.__dict__.pop('_constructor_index',None)
        old,one=Counter(),Counter()
        baseline=cold.seed(hint,np.random.default_rng(seed),constructor='compact',stats=old)
        reused=a.seed(hint,np.random.default_rng(seed),constructor='compact',stats=one)
        if not np.array_equal(baseline,reused):raise AssertionError('Changed deterministic start.')
        rows.append(dict(seed=seed,score=p.validate(reused.tolist())['matched_edges'],cold=dict(old),warm=dict(one)))
    atomic_json(output,dict(input=p.fingerprint,records=rows))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tiles',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--seeds',type=int,nargs='+',default=list(range(35390,35422)))
    ap.add_argument('--work',type=int,default=2000)
    ap.add_argument('--contours',action='store_true')
    ap.add_argument('--warm',action='store_true')
    ap.add_argument('--policies',nargs='+',default=('released','forward','hall','matching','compact'),choices=('released','forward','hall','matching','compact'))
    args=ap.parse_args()
    p=SearchProblem(read_json(args.tiles))
    if args.warm:return repeated_starts(p,args.output,args.seeds)
    if args.contours:
        return contours(p,args.output,args.work,args.policies)
    model=BoardModel(p)
    a=Assemblies(model)
    rows=[]
    for seed in args.seeds:
        rng=np.random.default_rng(seed)
        hint=fresh_start(model,rng)
        witness=a.seed(fresh_start(model,rng),rng)
        instances=[('unrestricted',a)]
        if seed%2==0:
            instances.append(('two-extra',conditional(a,hint,witness,rng,2)))
        if seed%4==0:
            instances.append(('five-extra',conditional(a,hint,witness,rng,5)))
        for name,obj in instances:
            item=dict(seed=seed,domains=name,known_witness_score=int(model.matches(witness).sum()),policies={})
            obj.model.legal_board(witness)
            item['policies']=measured(p,obj,hint,seed,args.work,args.policies)
            rows.append(item)
            atomic_json(args.output,dict(input=p.fingerprint,budget=args.work,records=rows))
            print(name,seed,[(k,v['score'],v['work'].get('pose_nodes')) for k,v in item['policies'].items()],flush=True)


if __name__=='__main__':
    main()
