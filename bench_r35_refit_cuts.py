"""Compare exact re-use of universally conditioned endpoint partitions on fresh phase objectives."""
from argparse import ArgumentParser
from pathlib import Path
from r32_gold_loss import read_json
from r32_fresh_phases import fit,phase_cube
from r32_gold_bonds import inventory
from search_problem import SearchProblem
from io_utils import atomic_json

if __name__=='__main__':
    cli=ArgumentParser(description=__doc__)
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args()
    p=SearchProblem(read_json('../data/tiles.json'))
    req=inventory(p).required
    records=[]
    from r35_endpoint_partitions import pinned_start_pool
    for width in (0,2,4):
        pool={}
        for seed in (4,5,6,7,11,19,31,51,71,81,91,101):
            pins=phase_cube(p,width,seed%(1<<width))
            arms=[]
            for name,store in [('cold',None),('shared',pool),('fixed-singleton',pinned_start_pool(p))]:
                out=fit(p,req,pins,(),seed=seed,effort=.15,endpoint_pool=store)
                row={k:v for k,v in out.items() if k!='phases'}
                if 'phases' in out:
                    from r29_gold_phase import EndpointCuts
                    from r30_gold import _groups
                    g=EndpointCuts(p)
                    assert all(out['phases'][i]==v for i,v in pins.items())
                    assert len(_groups(edge for t,z in enumerate(out['phases']) for edge in g.strands[t][z]))==1
                arms.append(dict(arm=name,**row))
            records.append(dict(width=width,seed=seed,cold=arms[0],shared=arms[1],fixed=arms[2]))
            atomic_json(args.output,records)
            print(width,seed,[(a['status'],a['calls'],a['cuts'],a['reused_cuts']) for a in arms],flush=True)
