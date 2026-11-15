"""Compare fair center-group turns and serial work to a planted full-edge witness."""
from contextlib import closing
import argparse,json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory

from r30_multicenter import run
from r34_centers import search,select
from r34_cycle_orders import CycleOrders
from search_problem import SearchProblem
from search_fixtures import planted_gold


def fixture():
    tiles,geo,gold=planted_gold(18,2,4)
    p=SearchProblem(tiles,geo,fixed={i:int(gold[i]) for i in range(22,len(gold))})
    family=(2,4);order=CycleOrders(family)[0]
    first=list(gold);other=list(gold)
    for a in range(6,22,2):first[a],first[a+1]=first[a+1],first[a]
    for i in range(6):other[i]=gold[order[i]]
    for i in range(22):other[i]=3*(other[i]//3)+(other[i]%3+1)%3
    return p,[first,other],family


def compare(tmp):
    p,boards,family=fixture();target=len(p.edges)-18
    args=dict(family=family,target=target,policy='score',limit=2)
    kwargs=dict(start=0,stop=4,page=1)
    db=Path(tmp)/'direct.db'
    cand,groups,info=select(p,boards,db,family,target,2,'score')
    assert [len(g) for g in groups]==[1,1],info
    traces=[];outcomes=[]
    for group in groups:
        for lo in range(4):
            rec=run(p,[cand[group[0]]['board']],db,families=(family,),target=target,start=lo,stop=lo+1,page=1)
            traces.append(rec['best']['score'] if rec['best'] else None)
        outcomes.append(rec)
    assert len(p.edges) in traces
    first_gold=traces.index(len(p.edges))+1
    baseline=[(r['scope'],r['best'],r['covered_supports']) for r in outcomes]
    result={'serial_turn_to_planted_gold':first_gold,'target':target,'max_edges':len(p.edges),'initial_scores':[r['score'] for r in info['selected']]}
    for policy in ('fair','yield'):
        target_db=Path(tmp)/(policy+'.db')
        from unittest.mock import patch
        from r30_multicenter import SharedSupports
        with patch('r30_multicenter.SharedSupports',wraps=SharedSupports) as built:
            trial=search(p,boards,target_db,schedule=policy,group_pages=1,**args,**kwargs)
        assert built.call_count==2
        final=[(r['scope'],r['best'],r['covered_supports']) for r in trial['searches']]
        assert final==baseline,(final,baseline)
        assert p.validate(trial['best']['codes'])['valid']
        event=next(record for record in trial['schedule']['improvements'] if record['score']==len(p.edges))
        assert event['turn']<first_gold,(event,first_gold,traces,trial['schedule'])
        warm=search(p,boards,target_db,schedule=policy,group_pages=1,**args,**kwargs)
        assert all(r['new_pages']==0 for r in warm['searches'])
        result[policy]=dict(**trial['schedule'],group_graph_builds=built.call_count,warm=[r['new_pages'] for r in warm['searches']],covered=[r['covered_supports'] for r in trial['searches']])
    return result


def main():
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('--output',type=Path);args=cli.parse_args()
    scratch=Path('audit/r35');scratch.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=scratch) as tmp:report=compare(tmp)
    if args.output:args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    else:print(json.dumps(report))


if __name__=='__main__':main()
