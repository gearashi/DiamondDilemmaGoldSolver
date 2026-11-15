"""Compare loss reachability on the outside syndrome alone and including admissible anchor prices."""
import argparse
import json
from math import inf
from pathlib import Path

from r32_gold_bonds import inventory
from r32_gold_loss import read_json,reference_groups
from r32_gold_syndrome import OutsideSyndromes
from search_problem import SearchProblem


def compare(tiles):
    problem=SearchProblem(read_json(tiles))
    docs=Path(__file__).parent/'docs'
    ref,_,groups=reference_groups(problem,read_json(docs/'R29_INCUMBENT.json')['codes'])
    required=inventory(problem).required
    records=[]
    for cap,e,start,stop in [(14,4,0,6903),(16,6,7044800,7052992),(17,4,0,6903),(17,5,0,65536),
                             (18,5,33361,33617)]:
        expected=None
        for weighted in (False,True):
            q=OutsideSyndromes(problem,ref,groups,required,pair_bounds=2,weighted_reach=weighted)
            trace=[]
            def note(indices,*args):
                trace.append(tuple(sorted(indices)))
                return False
            q.joint=note
            out=q.indexed(cap,e,inf,start,stop)
            if expected is None:expected=trace
            if expected!=trace or out['next_rank']!=stop:raise AssertionError('changed arrival order')
            primary=q._suffix_cache[1].weighted_reach.get(0)
            records.append(dict(cap=cap,errors=e,start=start,stop=stop,weighted=weighted,
                                prefix_work=out['work'].get('prefixes',0),joint=len(trace),
                                outside_states=q._reachable.states_built if q._reachable else 0,
                                weighted_states=primary.states_built if primary is not None else 0))
            print(cap,e,weighted,records[-1]['prefix_work'],len(trace),flush=True)
    return records


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiles',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.write_text(json.dumps(compare(args.tiles),indent=2)+'\n',encoding='utf8')
