"""Crash and resume both neighborhood algorithms and compare their complete physical graph."""
import argparse
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

from io_utils import atomic_json
from r27_incumbent import BoardModel
from r28_exchange import ExchangeBound,read_json
from r29_plateau import SwapGraph,walk
from r35_plateau_pages import advance,schema
from search_problem import SearchProblem


def worker(args):
    problem=SearchProblem(read_json(args.tiles))
    center=read_json(args.incumbent)['codes']
    model=BoardModel(problem)
    records=args.directory/(args.mode+'-events.jsonl')
    dbpath=args.directory/(args.mode+'.sqlite')
    def record(event):
        with records.open('a',encoding='utf8') as f:
            print(json.dumps(event),file=f)
    with closing(sqlite3.connect(dbpath)) as db:
        graph=SwapGraph(db,model)
        schema(db)
        db.execute('CREATE TABLE IF NOT EXISTS walk_legacy(board TEXT PRIMARY KEY,score INTEGER,seed TEXT)')
        key=graph.key(center)
        graph.optimum(key,center,0)
        db.commit()
        raw=graph.optimum
        stars=ExchangeBound.bounds
        def bound(self,cells,order,*a,**kw):
            out=stars(self,cells,order,*a,**kw)
            record(dict(stars=len(cells)))
            return out
        ExchangeBound.bounds=bound
        count=0
        def optimum(child,board,floor,*a,**kw):
            nonlocal count
            previous=graph.work['solved']
            item=raw(child,board,floor,*a,**kw)
            if graph.work['solved']>previous:
                count+=1
                record(dict(solved=child))
                if args.kill and count==args.kill:
                    os._exit(91)
            return item
        graph.optimum=optimum
        if args.mode=='released':
            db.execute('BEGIN IMMEDIATE')
            list(graph.neighbors(key,center,args.floor))
            db.commit()
        else:
            graph.work['swap_candidates']=0
            advance(graph,dbpath,key,center,args.floor,ttl=1)
        edges=list(db.execute('SELECT neighbor FROM walk_edges WHERE board=? ORDER BY neighbor',(key,)))
        verified=[]
        for child, in edges:
            raw=db.execute('SELECT result FROM optima WHERE key=?',(child,)).fetchone()
            item=json.loads(raw[0])
            graph.check(child,item['codes'],item['score'])
            verified.append((child,item['score'],item['components']))
        atomic_json(args.directory/(args.mode+'-result.json'),dict(edges=len(edges),physical_sha256=hashlib.sha256(json.dumps(verified).encode()).hexdigest(),
                    qualifying=sum(1 for child,score,components in verified if score>=args.floor),work=graph.work))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tiles',type=Path,required=True)
    ap.add_argument('--incumbent',type=Path,required=True)
    ap.add_argument('--directory',type=Path,required=True)
    ap.add_argument('--floor',type=int,default=207)
    ap.add_argument('--mode',choices=('released','fenced'))
    ap.add_argument('--kill',type=int,default=8)
    args=ap.parse_args()
    if args.mode:return worker(args)
    args.directory.mkdir(parents=True,exist_ok=True)
    if any(args.directory.iterdir()):raise ValueError('Use an empty directory for comparable crash experiments.')
    report={}
    for mode in ('released','fenced'):
        cmd=[sys.executable,'-B',__file__,'--tiles',str(args.tiles),'--incumbent',str(args.incumbent),
             '--directory',str(args.directory),'--mode',mode,'--floor',str(args.floor)]
        proc=subprocess.run(cmd+['--kill',str(args.kill)],check=False)
        if proc.returncode!=91:raise RuntimeError(('Expected actual forced process exit',proc.returncode))
        crash=read_events(args.directory,mode)
        with closing(sqlite3.connect(args.directory/(mode+'.sqlite'))) as db:
            high=db.execute('SELECT lowest FROM walk_bounds UNION ALL SELECT lowest FROM walk_vectors').fetchall()
            committed=db.execute('SELECT count(*) FROM optima').fetchone()[0]-1
            cursor=db.execute('SELECT next_pair FROM walk_band_cursor').fetchall()
        subprocess.run(cmd+['--kill','0'],check=True)
        both=read_events(args.directory,mode)
        solved=[row['solved'] for row in both if 'solved' in row]
        report[mode]=dict(crash_star_pairs=sum(x.get('stars',0) for x in crash),star_pairs_total=sum(x.get('stars',0) for x in both),
                          committed_at_crash=committed,closed_floor_at_crash=high,cursor_at_crash=cursor,
                          rotations_repeated=len(solved)-len(set(solved)),total_exact_rotations=len(solved),
                          final=read_json(args.directory/(mode+'-result.json')))
    for key in ('edges','physical_sha256','qualifying'):
        assert report['released']['final'][key]==report['fenced']['final'][key]
    atomic_json(args.directory/'comparison.json',report)
    print(json.dumps(report),flush=True)


def read_events(folder,mode):
    return [json.loads(x) for x in (folder/(mode+'-events.jsonl')).read_text().splitlines()]


if __name__=='__main__':
    main()
