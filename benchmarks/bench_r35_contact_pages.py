"""Check that additional original rank exclusions preserve all 208 endpoints in complete pages."""
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import argparse
import json

from io_utils import atomic_json
from r28_exchange import read_json
from r30_multicenter import run
from r32_route_ranges import PrefixRanges
from search_problem import SearchProblem


def compare(tiles,out):
    problem=SearchProblem(read_json(tiles))
    boards=[[3*t for t in row['pieces']] for row in read_json('docs/R29_EXCHANGE_CENTERS.json')['centers']]
    selections=read_json('docs/R35_ROUTE_CONTACT_HELDOUT.json')['trials']
    candidates=[x for x in selections if x['variants']['True']['open_ranks']<x['variants']['False']['open_ranks']]
    rows=[]
    with TemporaryDirectory(dir=out.parent) as folder:
        for x in candidates[:2]+candidates[-1:]:
            score={}
            for mode in ('old','new'):
                context=(patch('r32_route_ranges.PrefixRanges',side_effect=lambda *args:PrefixRanges(*args,joint=False))
                         if mode=='old' else nullcontext())
                with context:
                    actual=run(problem,boards,Path(folder)/(mode+str(x['start'])+'.sqlite'),start=x['start'],
                               stop=x['start']+x['ranks'],page=256,families=[tuple(x['family'])])
                assert actual['range_finished']
                score[mode]=dict(work=actual['work'],best=actual['best'])
            assert score['old']['best']==score['new']['best']
            rows.append(dict(start=x['start'],ranks=x['ranks'],family=x['family'],results=score))
            atomic_json(out,rows)
            print(x['start'],[(m,r['work'].get('first_rank_interval'),r['work'].get('raw_center_endpoints')) for m,r in score.items()],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tiles',default='../data/tiles.json')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    compare(a.tiles,a.output)
