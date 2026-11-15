"""Isolated pre/post native propagation and avoided CUDA lane initializations."""
from collections import Counter
from pathlib import Path
from time import perf_counter
import argparse
import json

from bench_r35_pose_frontiers import RotatedFacts, anchored, observe, saved_stacks
from gpu_domain_round24 import DomainCUDA
from search_problem import SearchProblem


def native_cases(p, cases, facts, batch, options):
    counts=Counter()
    samples=[]
    all_outputs={}
    begin=perf_counter()
    for start in range(0,len(cases),batch):
        group=cases[start:start+batch]
        engine=DomainCUDA(p,[ds for _,ds in group],target='gold',**options)
        try:
            done=False
            while not done:
                state,done=engine.propagate_root(work=4096)
                counts['dispatches']+=1
            base=engine.offsets['domain']
            data=engine.state[base:base+engine.n*engine.words].get()
            h=engine.state[:20].get()
            for k,(label,_) in enumerate(group):
                ds=None if state[k]==2 else [sum(int(data[c*engine.words+j,k]) << (32*j) for j in range(engine.words)) for c in range(engine.n)]
                observe(counts,samples,label,ds,facts)
                all_outputs[label]=ds
                counts['native_removals']+=int(h[12,k])
                counts['native_nodes']+=int(h[11,k])
        finally:
            engine.close()
    return dict(counts=dict(counts),examples=samples,seconds=perf_counter()-begin),all_outputs


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tiles',type=Path,default=Path('../data/tiles.json'))
    ap.add_argument('--checkpoints',type=Path,default=Path('../runtime/searches'))
    ap.add_argument('--cpu',type=Path,default=Path('audit/r35/pose-frontier-cpu.json'))
    ap.add_argument('--batch',type=int,choices=[1],default=1,help='Cases overlap, so each has a dedicated native owner.')
    ap.add_argument('--natural',type=int,default=4)
    ap.add_argument('--anchors',type=int,default=24)
    ap.add_argument('--symmetry',action='store_true')
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    p=SearchProblem(json.loads(args.tiles.read_bytes()),physical_gold_support=True)
    f=RotatedFacts(p)
    full=[(1<<(3*p.n))-1]*p.n
    original=f.record['codes']
    groups={'fresh':[('root',full)],
            'historical':list(saved_stacks(args.checkpoints))[::max(1,51//args.natural)],
            'singletons':[(str(c),anchored(full,[(c,original[c])])) for c in range(args.anchors)]}
    options=dict(event_propagation=True,cooperative=True)
    cpu=json.loads(args.cpu.read_bytes())['cpu']
    # Positive-unit and negative-case controls selected before launching CUDA.
    import ast
    subs=[ast.literal_eval(r['label']) for r in cpu['near_unit']['examples'][:8]]
    groups['near_unit']=[(str(row),anchored(full,[(c,original[c]) for c in row])) for row in subs]
    if args.symmetry:
        groups['transformed']=[]
        for idx,pairs in enumerate(f.images):
            for c in [5,6,9,112]:
                site,bit=pairs[c]
                groups['transformed'].append((f'{idx}-{c}',anchored(full,[(site,bit.bit_length()-1)])))
    result={}
    for name,cases in groups.items():
        if not cases:continue
        row,outs=native_cases(p,cases,f,args.batch,options)
        print(name,row['counts'],round(row['seconds'],2),flush=True)
        result[name]=row
        args.output.write_text(json.dumps(result,indent=2),encoding='utf8')

if __name__=='__main__':main()
