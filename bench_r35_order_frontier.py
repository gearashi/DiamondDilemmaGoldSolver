"""Measure real repeated order classifications after an interrupted support page."""
import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from unittest.mock import patch

from r28_exchange import read_json
from r30_multicenter import run
from r32_route_store import RootCoverage
from r34_cycle_orders import CycleOrders
from search_problem import SearchProblem
from search_fixtures import planted_gold


class SimulatedStop(Exception):
    pass


def measured(problem,boards,db,settings,quantum,cut=None):
    original=RootCoverage.covered
    observed=0
    def counted(self,*args):
        nonlocal observed
        if observed==cut:raise SimulatedStop
        observed+=1
        return original(self,*args)
    with patch.object(RootCoverage,'covered',counted):
        try:result=run(problem,boards,db,order_quantum=quantum,**settings)
        except SimulatedStop:
            if cut is None:raise
            result=None
    return result,observed


def planted_case():
    tiles,geo,codes=planted_gold(29,1,4)
    p=SearchProblem(tiles,geo);family=(2,2,2,3)
    options=CycleOrders(family);cells=list(range(sum(family)))
    boards=[]
    for rank in (0,1):
        board=list(codes)
        for i,c in enumerate(cells):board[c]=codes[options[rank][i]]
        boards.append(board)
    return p,boards,family


def compare(p,boards,family,target,rank,tmp,cut):
    settings=dict(families=(family,),target=target,start=rank,stop=rank+1,page=1,filters=())
    out={}
    reference=None
    for q in (0,128):
        db=str(Path(tmp)/f'{len(boards)}-{q}.db')
        partial,first=measured(p,boards,db,settings,q,cut)
        assert partial is None
        with closing(sqlite3.connect(db)) as con:
            has=con.execute("SELECT 1 FROM sqlite_master WHERE name='r35_order_cursor'").fetchone()
            rows=con.execute("SELECT done,length(payload) FROM r35_order_cursor").fetchall() if has else []
            assert con.execute('SELECT count(*) FROM r30_pages').fetchone()[0]==0
        result,after=measured(p,boards,db,settings,q)
        assert result['range_finished']
        with closing(sqlite3.connect(db)) as con:
            stored=con.execute("SELECT work,best FROM r30_pages").fetchone()
            deps=con.execute('SELECT owner,family,payload FROM r35_page_dependencies ORDER BY owner,family').fetchall()
            counts=json.loads(stored[0])
            if q:assert con.execute('SELECT count(*) FROM r35_order_cursor').fetchone()[0]==0
        first_cuts={k:v for k,v in counts.items() if k.startswith('first_') or k in ('raw_center_endpoints','hits','illegal')}
        if result['best']:
            assert p.validate(result['best']['codes'])['matched_edges']==result['best']['score']
        evidence=(first_cuts,result['best'],deps)
        if q:assert evidence==reference
        else:reference=evidence
        warm,work=measured(p,boards,db,settings,q)
        assert work==0 and warm['best']==result['best']
        out[str(q)]=dict(pre_stop=first,post_stop=after,total_checks=first+after,reused=result['work'].get('order_rows_reused',0),
                         aggregate_cuts=first_cuts,partial=rows,typed_dependency_rows=len(deps),
                         witness_score=result['best']['score'] if result['best'] else None)
    assert out['0']['total_checks']-out['128']['total_checks']==sum(n for n,_ in out['128']['partial'])>0
    return out


def main():
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('--tiles',default='data/tiles.json')
    cli.add_argument('--output',type=Path);args=cli.parse_args()
    p=SearchProblem(read_json(args.tiles));board=read_json('docs/R29_INCUMBENT.json')['codes']
    scratch=Path('audit/r35');scratch.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=scratch) as tmp:
        report={'original_forced_endpoint_comparison':compare(p,[board],(2,2,2,3),208,0,tmp,587)}
        result,elig=measured(p,[board],Path(tmp)/'default.db',dict(families=((2,2,2,3),),target=208,start=0,stop=1,page=1),128)
        report['original_default']=dict(classified_orders=elig,work=result['work'])
        q,boards,family=planted_case()
        report['heldout_two_centers']=compare(q,boards,family,len(q.edges),0,tmp,3301)
    if args.output:args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    else:print(json.dumps(report))


if __name__=='__main__':main()
