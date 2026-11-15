"""Revisit naturally reached original Gold search subtrees with identical decisions and budgets."""
from pathlib import Path
from random import Random
import argparse,json
from bench_r35_pose_frontiers import Compiled
from r35_pose_frontiers import FrontierPoses
from search_problem import SearchProblem


def compare(p,compiled,seed,collect_steps,subtree_steps):
    full=[(1<<(3*p.n))-1]*p.n
    rng=Random(seed)
    rank=list(range(p.n));rng.shuffle(rank)
    codes=list(range(3*p.n));rng.shuffle(codes)
    receiver=FrontierPoses(p)
    scopes=[]
    seen=set()
    def run(ds,mode,budget):
        def visit(domains,path):
            assignments=[value.bit_length()-1 if value.bit_count()==1 else -1 for value in domains]
            if p.partial_loop_impossible(assignments):return dict(action='reject')
            if mode in ('sample','apply'):
                changed,info=receiver.prune(domains)
                if mode=='sample' and info and (info['over'] or info['removed']) and tuple(domains) not in seen:
                    seen.add(tuple(domains))
                    scopes.append((domains,[(c,v.bit_length()-1) for c,v in path]))
                if mode=='apply' and info:
                    if info['over']:return dict(action='reject')
                    domains=changed
            c=min((i for i,d in enumerate(domains) if d.bit_count()>1),key=lambda i:(domains[i].bit_count(),rank[i]),default=None)
            if c is None:
                board=[v.bit_length()-1 for v in domains]
                return dict(action='solved',value=board) if p.validate(board)['valid'] else dict(action='reject')
            return dict(action='branch',domains=domains,cell=c,choices=[1<<v for v in codes if domains[c]>>v&1])
        ans=compiled.engine.run([dict(domains=ds,cell=-1,choices=[],next=0)],visit,max_steps=budget)
        return dict(status=ans['status'],stats={k:v for k,v in ans['stats'].items() if k in ('native_search_nodes','native_search_branches','native_fast_rejects','native_fast_arcs','native_search_transitions')})
    sample=run(full,'sample',collect_steps)
    records=[]
    for ds,path in scopes:
        base=run(ds,'off',subtree_steps)
        trial=run(ds,'apply',subtree_steps)
        if base['status']=='exhausted' and trial['status']=='solved':raise AssertionError('Contradictory finite result')
        if trial['status']=='exhausted' and base['status']=='solved':raise AssertionError('Contradictory finite result')
        records.append(dict(path=path,base=base,trial=trial))
    return dict(seed=seed,live_scopes_with_units=len(seen),sample=sample,paired=records)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tiles',type=Path,default=Path('../data/tiles.json'))
    ap.add_argument('--native',type=Path,default=Path('../.native'))
    ap.add_argument('--steps',type=int,default=4096)
    ap.add_argument('--subtree',type=int,default=512)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    p=SearchProblem(json.loads(args.tiles.read_bytes()),physical_gold_support=True)
    engine=Compiled(p,args.native)
    out=[]
    try:
        for seed in (0,1,17,89):
            row=compare(p,engine,seed,args.steps,args.subtree)
            out.append(row)
            print(seed,row['live_scopes_with_units'],[(x['base']['status'],x['base']['stats']['native_search_nodes'],x['trial']['stats']['native_search_nodes']) for x in row['paired']],flush=True)
            args.output.write_text(json.dumps(out,indent=2),encoding='utf8')
    finally:engine.close()

if __name__=='__main__':main()
