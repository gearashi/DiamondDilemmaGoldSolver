"""Replay independently priced route pools in both directions and with overlapping root workers."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from threading import Event
from unittest.mock import patch

from r28_exchange import read_json
from r31_route_cursor import advance
from r31_routes import search_roots
from r32_route_pool import evaluate
from search_problem import SearchProblem


def parallel_roots(run,path):
    ready,release=Event(),Event()
    def hold_first(*args,**kwargs):
        if not ready.is_set():
            ready.set()
            if not release.wait(30):raise RuntimeError('Independent root did not advance')
        return advance(*args,**kwargs)
    with patch('r31_route_cursor.advance',side_effect=hold_first),ThreadPoolExecutor(2) as threads:
        first=threads.submit(run,path,start=0,stop=1)
        try:
            if not ready.wait(30):raise RuntimeError('First root did not start')
            with closing(sqlite3.connect(path,timeout=1)) as peer:
                peer.execute('BEGIN IMMEDIATE');peer.rollback()
            second=threads.submit(run,path,start=0,stop=2,max_pages=1)
            other=second.result(timeout=30)
        finally:release.set()
        one=first.result(timeout=30)
    tail=run(path,start=0,stop=2)
    return one,other,tail


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--tiles',default='data/tiles.json');cli.add_argument('--output',type=Path)
    args=cli.parse_args();p=SearchProblem(read_json(args.tiles));board=read_json('docs/R29_INCUMBENT.json')['codes']
    scratch=Path('audit/r35');scratch.mkdir(parents=True,exist_ok=True)
    report=[]
    with TemporaryDirectory(dir=scratch) as tmp:
        for ix,(family,cells) in enumerate([((2,2,3),[18,30,42,44,51,62,118,122]),((2,2,2,2),list(range(8)))]):
            def root(path,**kw):return search_roots(p,board,family,208,path,pool=cells,quantum=100000,catalogue=False,**kw)
            def pool(path):return evaluate(p,[board],cells,family,208,database=path)
            def db(name):return str(Path(tmp)/f'{ix}-{name}.db')
            direct=root(db('plain'));region_after=pool(db('plain'))
            region_first=pool(db('other'));root_after=root(db('other'))
            assert all(r['best'] is None for r in (direct,region_after,region_first,root_after))
            assert direct['complete'] and root_after['complete'] and region_first['complete'] and region_after['complete']
            serial=root(db('serial'),start=0,stop=2)
            workers=parallel_roots(root,db('dual'))
            steps=sum(w['work'].get('cursor_steps',0) for w in workers)
            assert steps==serial['work']['cursor_steps'] and workers[-1]['range_finished']
            with patch('r31_routes.ConnectedJoin',side_effect=AssertionError('rebuilt complete roots')):
                warm=root(db('dual'),start=0,stop=2)
            report.append(dict(family=family,pool=cells,roots=direct['completed_roots'],root_first=direct['work'],
                        region_after=region_after['work'],region_first=region_first['work'],root_after=root_after['work'],
                        serial_two=serial['work'],dual_two=[w['work'] for w in workers],warm=warm['work']))
            print(family,steps,region_after['work'].get('receipt_or_unowned_stars_avoided'),flush=True)
    if args.output:args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    else:print(json.dumps(report))


if __name__=='__main__':main()
