"""Measure exact pair and shortlist extensions on identical owned physical loss ranges."""
import argparse
import json
from math import inf
from pathlib import Path

from r32_gold_bonds import inventory
from r32_gold_loss import read_json,reference_groups
from r32_gold_syndrome import OutsideSyndromes
from search_problem import SearchProblem


def compare(tiles, five_stop, widths):
    problem=SearchProblem(read_json(tiles))
    docs=Path(__file__).parent/'docs'
    ref,_,g=reference_groups(problem,read_json(docs/'R29_INCUMBENT.json')['codes'])
    req=inventory(problem).required
    result=[]
    for cap,errors,start,stop in ((14,4,0,6903),(15,5,60000,68192),(16,6,7044800,7052992),
                                  (17,4,0,6903),(17,5,0,five_stop),(18,5,33361,33617)):
        for paired in widths:
            q=OutsideSyndromes(problem,ref,g,req,pair_bounds=paired)
            hits=[]
            def observed(row, deadline):
                hits.append(tuple(sorted(row)))
                return False
            answer=q.indexed(cap,errors,inf,start,stop,accept_errors=observed)
            if not paired:
                expected=sorted(hits)
            if expected!=sorted(hits) or answer['next_rank']!=stop:
                raise AssertionError('group table changed admissible outcomes')
            result.append(dict(cap=cap,errors=errors,start=start,stop=stop,paired=paired,hits=hits,
                               work=answer['work'],index_work=dict(q.index_work)))
            print(cap,errors,paired,'tails',answer['work'].get('indexed_tails',0),'joint',answer['work'].get('joint_queries',0),'hits',len(hits),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiles',required=True)
    parser.add_argument('--widths',type=int,nargs='+',default=[0,2,3,10])
    parser.add_argument('--five-stop',type=int,default=65536)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.write_text(json.dumps(compare(args.tiles,args.five_stop,args.widths),indent=2)+'\n',encoding='utf8')
