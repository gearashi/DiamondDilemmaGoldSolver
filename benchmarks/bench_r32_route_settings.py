"""Compare ranked page sizes and devices on identical disjoint original ranges."""
import argparse
import json
from pathlib import Path
from statistics import median
from time import monotonic
from r28_exchange import read_json
from r30_multicenter import run
from search_problem import SearchProblem


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--tiles',default='data/tiles.json')
    cli.add_argument('--output',required=True)
    cli.add_argument('--cuda',action='store_true')
    args=cli.parse_args()
    p=SearchProblem(read_json(args.tiles))
    boards=[[3*t for t in item['pieces']] for item in read_json('docs/R29_EXCHANGE_CENTERS.json')['centers']]
    rows=[]
    for family,starts in (((2,2,3),(301415269975,200000000023)),((2,2,2,2),(5360481967048,500000000017))):
        for start in starts:
            configs=[(d,page) for d in (['cpu','cuda'] if args.cuda else ['cpu']) for page in (128,256,1024)]
            expected=None
            for backend,page in configs:
                elapsed=[]
                for repeat in range(2):
                    before=monotonic()
                    res=run(p,boards,':memory:',families=(family,),start=start,stop=start+2048,page=page,backend=backend)
                    elapsed.append(monotonic()-before)
                    assert res['range_finished'] and res['covered_supports']==2048
                    if expected is None:
                        expected=(res['best'],res['work']['raw_center_endpoints'])
                    assert expected==(res['best'],res['work']['raw_center_endpoints'])
                w=res['work']
                rows.append(dict(family=family,start=start,device=backend,page=page,seconds=round(median(elapsed),4),
                                 raw=w['raw_center_endpoints'],ranges_skipped=w.get('supports_before_generation',0),
                                 probes=w.get('interval_probes',0),assignment_solves=w.get('support_assignments',0),
                                 exact=w.get('exact_solved',0),sqlite_pages=res['new_pages'],best=res['best']))
    Path(args.output).write_text(json.dumps(dict(fingerprint=p.fingerprint,rows=rows),indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
