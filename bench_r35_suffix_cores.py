"""Measure verified integer-core ordinal cuts and secondary bucket union reuse."""
from math import inf
from pathlib import Path
import argparse
import json

from r32_gold_bonds import inventory
from r32_gold_loss import read_json,reference_groups
from r32_gold_syndrome import OutsideSyndromes
from r33_gold_refine import IntegerPatternFilter
from search_problem import SearchProblem


def compare(tiles,five_stop):
    docs=Path(__file__).parent/'docs'
    p=SearchProblem(read_json(tiles))
    ref,_,g=reference_groups(p,read_json(docs/'R29_INCUMBENT.json')['codes'])
    req=inventory(p).required
    base=OutsideSyndromes(p,ref,g,req)
    checker=IntegerPatternFilter(p,ref,base.groups,req,base.outside,17)
    for idx in read_json(docs/'R34_INTEGER_CORES_FINAL.json')['arrivals']:
        if checker(idx,inf):
            raise AssertionError('Archived integer-negative changed')
    if not checker.verified:
        raise AssertionError('Expected completed core evidence')
    rows=[]
    for cap,e,start,stop in ((14,4,0,6903),(16,6,7044800,7052992),(17,4,0,6903),(17,5,0,five_stop)):
        arms={}
        expected=None
        for tag,unions,early in [('cold-unions',False,False),('shared-unions',True,False),('early-cores',True,True)]:
            q=OutsideSyndromes(p,ref,g,req,reuse_bucket_unions=unions,prune_prefixes=False)
            hits=[]
            def record(idx,d):
                hits.append(sorted(idx))
                return False
            opts=dict(accept_errors=record)
            if cap==17:
                opts['reject_before_joint']=checker.excludes
                if early:
                    opts['verified_cores']=lambda:checker.verified
            elif early:
                continue
            out=q.indexed(cap,e,inf,start,stop,**opts)
            if expected is None:
                expected=hits
            if out['next_rank']!=stop or hits!=expected:
                raise AssertionError((out['next_rank'],stop,hits,expected))
            arms[tag]=dict(work=out['work'],index_work=dict(q.index_work),phase_arrivals=hits)
            print(cap,e,tag,out['work'].get('indexed_tails'),out['work'].get('integer_core_suffixes_avoided'),dict(q.index_work),flush=True)
        rows.append(dict(cap=cap,errors=e,start=start,stop=stop,arms=arms))
    return dict(fingerprint=p.fingerprint,checked_cores=len(checker.verified),cases=rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiles',required=True)
    parser.add_argument('--five-stop',type=int,default=65536)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.write_text(json.dumps(compare(args.tiles,args.five_stop),indent=2)+'\n',encoding='utf8')
