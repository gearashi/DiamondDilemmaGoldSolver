"""Phase-first and regional fresh-board searches for the Gold single-loop objective."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json

import numpy as np

from io_utils import atomic_json
from r27_incumbent import BoardModel
from r28_exchange import domain_binding, read_json
from r29_gold_geography import checked_domains
from r31_fresh import Assemblies, Engine, physical_bonds, result
from r31_sampling import ranked_start, run_seeds
from r32_fresh_phases import fit, lift_cube, phase_cube, SignedRepair
from r32_gold_bonds import inventory
from search_problem import SearchProblem


def _poses(model, members, bonds):
    """Construct every rigid pose of one assumed physical component in the current domains."""
    adj=defaultdict(list)
    for a,b in bonds:
        t,s=divmod(a,3)
        u,j=divmod(b,3)
        if t in members:
            adj[t].append((s,u,j))
            adj[u].append((j,t,s))
    root=min(members)
    rows=[]
    for c in range(model.n):
        for r in range(3):
            if not model.legal[c,3*root+r]:
                continue
            pos={root:(c,r)}
            cells={c}
            pending=[root]
            valid=True
            while pending and valid:
                t=pending.pop()
                i,rot=pos[t]
                for s,u,j in adj[t]:
                    dest,side=model.problem.board.neighbors[i][(s+rot)%3]
                    orientation=(side-j)%3
                    target=dest,orientation
                    if u in pos:
                        valid=pos[u]==target
                    elif dest in cells or dest<0 or not model.legal[dest,3*u+orientation]:
                        valid=False
                    else:
                        cells.add(dest)
                        pos[u]=target
                        pending.append(u)
                    if not valid:
                        break
            if valid and len(pos)==len(members):
                rows.append(tuple((i,3*t+r) for t,(i,r) in sorted(pos.items())))
    return rows


def speculate(model, assemblies, hint, phases, rng, merges=3):
    """Grow speculative assemblies with legal poses. Back out optional joins if packing stalls."""
    from r29_gold_phase import EndpointCuts
    inv=inventory(model.problem)
    data=EndpointCuts(model.problem)
    required=assemblies.bonds
    partners={s for edge in required for s in edge}
    choices=defaultdict(list)
    for a,b in sorted(inv.possible):
        if a in partners or b in partners:
            continue
        ma,ca=data.sides[a]
        _,cb=data.sides[b]
        if phases[a//3]^phases[b//3] == ca^cb^(ma.bit_count()%2):
            choices[a].append(b)
            choices[b].append(a)
    weights={tuple(sorted((a,b))):len(choices[a])*len(choices[b]) for a in choices for b in choices[a]}
    order=sorted(weights,key=lambda e:(weights[e],rng.random()))
    groups=[set(tiles) for tiles,_ in assemblies.groups]
    rows=[poses for _,poses in assemblies.groups]
    extra=[]
    history=[]
    tries=0
    for a,b in order:
        if len(extra)>=merges:
            break
        if a in partners or b in partners:
            continue
        ta,tb=a//3,b//3
        selected=[i for i,g in enumerate(groups) if ta in g or tb in g]
        members={ta,tb}.union(*(groups[i] for i in selected))
        if len(members)>9 or any({ta,tb}<=groups[i] for i in selected):
            continue
        tries+=1
        poses=_poses(model,members,required|set(extra)|{(a,b)})
        if not poses:
            continue
        history.append((list(groups),list(rows)))
        for i in reversed(selected):
            del groups[i],rows[i]
        groups.append(members)
        rows.append(poses)
        extra.append((a,b))
        partners.update((a,b))
    attempts=len(extra)
    while extra:
        guessed=object.__new__(Assemblies)
        guessed.model=model
        guessed.forbidden=assemblies.forbidden
        guessed.bonds=required|set(extra)
        guessed.groups=sorted(zip(groups,rows),key=lambda row:len(row[1]))
        guessed.tiles=set.union(set(),*groups)
        try:
            return guessed.seed(hint,rng,work=5000),dict(tried=tries,retained=len(extra),backed_out=attempts-len(extra))
        except ValueError:
            groups,rows=history.pop()
            extra.pop()
    return assemblies.seed(hint,rng),dict(tried=tries,retained=0,backed_out=attempts)


def fresh(problem, seed, *, rounds=32, cube_bits=2, refit=0, merges=0, cache=None, backend='cpu',
          replicas=512, dispatches=64, sampler_cache=None, effort=.15, contour=False, saved=None, save=None):
    if min(seed,rounds,refit,merges)<0 or effort<=0:
        raise ValueError('Invalid seed or repair configuration.')
    model=BoardModel(problem)
    assemblies=Assemblies(model)
    inv=inventory(problem)
    pins=phase_cube(problem,cube_bits,seed%(1<<cube_bits))
    rng=np.random.default_rng(seed)
    if saved:
        board=model.legal_board(saved['current'])
        rng.bit_generator.state=saved['rng']
        phases=saved['phases']
        best=saved['best']
        stats=Counter(saved['stats'])
        first=saved['cursor']
        if not assemblies.bonds <= physical_bonds(model,board):
            raise ValueError('Checkpoint violates mandatory Gold bonds.')
    else:
        board=ranked_start(model,seed)
        stats=Counter()
        if contour:
            from r32_fresh_skeleton import parity_seed
            skeleton = parity_seed(problem,seed,conditioned=contour=='conditioned')
            if 'codes' in skeleton:
                board = model.legal_board(skeleton['codes'])
            stats['contour_trials'] = skeleton['trials']
            stats['contour_assignments'] = skeleton['assignments']
        if backend!='cpu':
            rows,_,new=Engine(problem)._archive(seed,backend,replicas,dispatches,sampler_cache)
            stats['gpu_proposals']=new
            if rows:
                board=max((model.legal_board(r) for r in [board,*rows]),key=lambda r:int(model.matches(r).sum()))
        start=result(model,board)
        if start['gold']:
            return dict(**start,seed=seed,work=dict(stats))
        f=fit(problem,assemblies.bonds,pins,seed=seed,effort=effort)
        stats['phase_calls']+=f['calls']
        # Stalled fresh seeds can use the unrestricted fit. Only lift_cube publishes proved exclusions.
        if 'phases' not in f and cube_bits:
            f=fit(problem,assemblies.bonds,{0:0},seed=seed,effort=effort)
            stats['released_cube']+=1
            stats['phase_calls']+=f['calls']
        if 'phases' not in f:
            board=assemblies.seed(board,rng)
            return dict(**result(model,board),seed=seed,phase_status=f['status'],work=dict(stats))
        phases=f['phases']
        if merges:
            board,info=speculate(model,assemblies,board,phases,rng,merges)
            stats.update({'speculative_'+k:v for k,v in info.items()})
        else:
            board=assemblies.seed(board,rng)
        best=result(model,board)
        first=0
    signed=SignedRepair(model,phases,forbidden=inv.forbidden,database=cache)
    if first==0:
        best['signed'],best['forbidden_bonds']=signed.score(board)
        best['phases']=list(phases)
    if best['gold']:
        return dict(**best,seed=seed,work=dict(stats))
    protected=assemblies.protected(board)
    for step in range(first,rounds):
        if step and refit and step%refit==0:
            old,_=signed.score(board)
            # Always remove the seed's optional phase pins during refitting.
            good=physical_bonds(model,board)-inv.forbidden
            f=fit(problem,assemblies.bonds,{0:0},good,seed=seed+step,effort=effort,hint=phases)
            stats['refit_calls']+=f['calls']
            if f.get('kept',-1)>old:
                stats.update(signed.stats)
                phases=f['phases']
                signed=SignedRepair(model,phases,forbidden=inv.forbidden,database=cache)
                stats['improved_refits']+=1
        cells=np.setdiff1d(model.independent_set(board,rng),protected)
        board=signed.block(board,cells)
        item=result(model,board)
        item['signed'],item['forbidden_bonds']=signed.score(board)
        item['phases']=list(phases)
        key=lambda r:(r['gold'],r['signed'],-r['forbidden_bonds'],r['score'],-r['components'])
        if key(item)>key(best):
            best=item
        stats['rounds']+=1
        for threshold in (100,110,120,130,140,150,160):
            if item['signed']>=threshold:
                stats.setdefault('round_to_signed_'+str(threshold),step+1)
        if save and ((step+1)%8==0 or step+1==rounds or best['gold']):
            save(dict(current=board.tolist(),rng=rng.bit_generator.state,phases=phases,best=best,
                      stats=dict(stats+signed.stats),cursor=step+1))
        if best['gold']:
            break
    stats.update(signed.stats)
    stats['mandatory_bonds']=len(assemblies.bonds)
    if not assemblies.bonds<=physical_bonds(model,model.legal_board(best['codes'])):
        raise ArithmeticError('Signed repair lost a mandatory bond.')
    return dict(**best,seed=seed,work=dict(stats))


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('mode',choices=['fresh','cube','skeleton','slots'])
    cli.add_argument('--tiles',default='data/tiles.json')
    cli.add_argument('--request',type=Path,help='Optional JSON domains, region cell lists, or a complete comparison board.')
    cli.add_argument('--database',type=Path)
    cli.add_argument('--row-cache',type=Path)
    cli.add_argument('--output',type=Path)
    cli.add_argument('--seed',type=int,default=0)
    cli.add_argument('--starts',type=int,default=1)
    cli.add_argument('--rounds',type=int,default=32)
    cli.add_argument('--cube-bits',type=int,default=2)
    cli.add_argument('--refit',type=int,default=0)
    cli.add_argument('--merges',type=int,default=0)
    cli.add_argument('--backend',choices=['cpu','cuda','webgpu'],default='cpu')
    cli.add_argument('--replicas',type=int,default=512)
    cli.add_argument('--dispatches',type=int,default=64)
    cli.add_argument('--sampler-cache',type=Path)
    cli.add_argument('--effort',type=float,default=.15)
    cli.add_argument('--seconds',type=float,default=2)
    cli.add_argument('--nodes',type=int,default=25000)
    cli.add_argument('--limit',type=int,default=1)
    cli.add_argument('--target',type=int)
    cli.add_argument('--layers',type=int,nargs='+',default=[0])
    cli.add_argument('--lift',action='store_true')
    cli.add_argument('--contour',action='store_true',help='Sample a geographic odd-seam skeleton before placing tiles.')
    cli.add_argument('--conditioned-contour',dest='contour',action='store_const',const='conditioned',
                     help='Fill independent vertex colors to reach the required active-cell count.')
    args=cli.parse_args()
    p=SearchProblem(read_json(args.tiles))
    request=read_json(args.request) if args.request else {}
    domains=None if 'domains' not in request else tuple(sum(1<<v for v in set(row)) for row in request['domains'])
    ds=checked_domains(p,domains)
    for path in (args.database,args.row_cache,args.output):
        if path:
            path.parent.mkdir(parents=True,exist_ok=True)
    if args.mode=='fresh':
        if not args.database:
            cli.error('fresh search requires --database')
        if domains is not None:
            cli.error('use cube, slots or skeleton for restricted domains')
        opts={k:getattr(args,k) for k in ('rounds','cube_bits','refit','merges','backend','replicas','dispatches','effort','contour')}
        scope=hashlib.sha256(json.dumps(['r32-fresh-v2',domain_binding(BoardModel(p)),opts],sort_keys=True).encode()).hexdigest()
        def run(seed,state,save):
            return fresh(p,seed,**opts,cache=args.row_cache,sampler_cache=args.sampler_cache,saved=state,save=save)
        def progress(row):
            print(json.dumps({k:v for k,v in row.items() if k!='codes'}),flush=True)
        answer=run_seeds(args.database,scope,args.seed,args.starts,run,progress=progress,checkpointed=True,output=args.output)
    elif args.mode=='cube':
        pins=phase_cube(p,args.cube_bits,args.seed)
        required=inventory(p).required
        answer=lift_cube(p,required,pins,ds,args.seconds,args.database) if args.lift else fit(
            p,required,pins,physical_bonds(BoardModel(p),BoardModel(p).legal_board(request['codes']))
            if 'codes' in request else (),seed=args.seed,effort=args.effort)
        answer['pins']=pins
    elif args.mode=='skeleton':
        from r32_fresh_geometry import skeletons, skeleton_domains
        if args.contour:
            from r32_fresh_skeleton import parity_seed
            sample=parity_seed(p,args.seed,domains=ds,trials=args.nodes,conditioned=args.contour=='conditioned')
            answer=dict(sample=sample,patterns=[sample['patterns']] if 'patterns' in sample else [],complete=False)
        else:
            answer=skeletons(p,ds,max_nodes=args.nodes,limit=args.limit)
        if args.lift:
            required=inventory(p).required
            answer['lifts']=[]
            for row in answer['patterns']:
                item=lift_cube(p,required,{0:0},skeleton_domains(p,ds,row),args.seconds,args.database)
                answer['lifts'].append(item)
                if item['status']=='GOLD':
                    break
    else:
        from r32_fresh_geometry import projected
        answer=projected(p,ds,layers=args.layers,target=args.target,regions=request.get('regions',()),effort=args.effort)
    if args.mode!='fresh' and args.output:
        atomic_json(args.output,answer)
    print(json.dumps(answer))


if __name__=='__main__':
    main()
