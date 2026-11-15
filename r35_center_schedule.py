"""Interleave finite page quotas without treating scheduler state as search coverage."""
from collections import Counter
from contextlib import closing
import hashlib
import json
from math import comb
import sqlite3
from time import monotonic


def interleave(problem,candidates,groups,database,family,target,settings,runner,*,quota=2,policy='fair'):
    if type(quota) is not int or quota<1 or policy not in ('fair','yield'):
        raise ValueError('Invalid center group schedule.')
    from r27_incumbent import BoardModel
    from r28_exchange import domain_binding
    from r34_rank_leases import covered_end
    model=BoardModel(problem)
    binding=domain_binding(model)
    start=settings.get('start',0)
    limit=comb(len(model.free),sum(family))
    stop=settings.get('stop')
    if stop is None:stop=limit
    page=settings.get('page',64);max_pages=settings.get('max_pages');seconds=settings.get('seconds',0)
    if not 0<=start<=stop<=limit or page<1 or seconds<0 or (max_pages is not None and max_pages<0):
        raise ValueError('Invalid ranked group limits.')
    states=[]
    for group in groups:
        handle=hashlib.sha256(json.dumps([binding,sorted(group),sorted(family),target]).encode()).hexdigest()
        physical=sorted(set(tuple(map(int,candidates[k]['board']//3)) for k in group))
        expected=hashlib.sha256(json.dumps([1,binding,physical,(tuple(sorted(family)),),target]).encode()).hexdigest()
        states.append(dict(handle=handle,expected=expected,cursor=start,pages=0,elapsed=0,work=Counter(),turns=0,result=None))
    trace=[];improvements=[];best_score=-1;turn_count=0
    with closing(sqlite3.connect(database,timeout=90,isolation_level=None)) as db:
        db.execute("""CREATE TABLE IF NOT EXISTS r35_center_turns(
            handle TEXT PRIMARY KEY,last_turn INTEGER,unique_orders INTEGER,orders INTEGER,ranked_scope TEXT)""")
        for state in states:
            db.execute("INSERT OR IGNORE INTO r35_center_turns VALUES(?,0,0,0,NULL)",(state['handle'],))
        active=set(range(len(states)))
        idle=set()
        while active:
            with db:
                db.execute('BEGIN IMMEDIATE')
                clock=db.execute('SELECT coalesce(max(last_turn),0) FROM r35_center_turns').fetchone()[0]+1
                observed={i:db.execute('SELECT last_turn,unique_orders,orders,ranked_scope FROM r35_center_turns WHERE handle=?',
                                        (states[i]['handle'],)).fetchone() for i in active}
                # One discretionary turn per rotation; least-recently-served turns
                # continue to bound waiting even when known yields are very different.
                extra=policy=='yield' and clock%(len(states)+1)==0 and all(o[2]>0 for o in observed.values())
                if extra:
                    ix=max(active,key=lambda i:(observed[i][1]/observed[i][2],-observed[i][0],-i))
                else:
                    ix=min(active,key=lambda i:(observed[i][0],i))
                state=states[ix]
                db.execute('UPDATE r35_center_turns SET last_turn=? WHERE handle=?',(clock,state['handle']))
            cursor=state['cursor']
            known_scope=observed[ix][3]
            has_pages=db.execute("SELECT 1 FROM sqlite_master WHERE name='r30_pages'").fetchone()
            if known_scope==state['expected'] and has_pages:cursor=covered_end(db,'r30_pages',known_scope,cursor,stop)
            allocation=min(quota,max_pages-state['pages']) if max_pages is not None else quota
            budget=seconds-state['elapsed'] if seconds else 0
            end=min(stop,cursor+quota*page)
            kw=dict(settings,start=cursor,stop=end,max_pages=max(0,allocation),seconds=max(budget,1e-6) if seconds else 0)
            before=monotonic()
            result=runner(problem,[candidates[k]['board'] for k in groups[ix]],database,families=(family,),target=target,**kw)
            state['elapsed']+=monotonic()-before
            state['turns']+=1;state['pages']+=result['new_pages'];state['work'].update(result['work'])
            if len(trace)<128:trace.append(ix)
            state['result']=result
            turn_count+=1
            witness=result['best']
            if witness and witness['score']>best_score:
                best_score=witness['score']
                event=dict(group=ix,turn=turn_count,ranked_pages=sum(g['pages'] for g in states),score=best_score)
                improvements.append(event)
                if settings.get('progress'):settings['progress'](dict(event='center_group_best',**event))
            if result['complete']:cursor=stop
            elif result['range_finished']:cursor=end
            if db.execute("SELECT 1 FROM sqlite_master WHERE name='r30_pages'").fetchone():
                cursor=covered_end(db,'r30_pages',result['scope'],cursor,stop)
            state['cursor']=cursor
            raw=result['work'].get('raw_center_endpoints',0)
            owned=result['work'].get('first_owned',0)+result['work'].get('first_prior_page',0)
            db.execute('UPDATE r35_center_turns SET unique_orders=unique_orders+?,orders=orders+?,ranked_scope=? WHERE handle=?',
                       (max(0,raw-owned),raw,result['scope'],state['handle']))
            if cursor>=stop or (max_pages is not None and state['pages']>=max_pages) or (seconds and state['elapsed']>=seconds):
                active.remove(ix)
            elif result['new_pages']==0 and not result['range_finished']:
                # Other processes may own every rank in this interval. Let this
                # invocation return bounded work, or wait if it was unlimited.
                if max_pages is not None:
                    active.remove(ix)
                else:idle.add(ix)
                if active and active<=idle:
                    from r34_rank_leases import RankedLeases
                    RankedLeases(db,'support',result['scope']).wait(cursor,monotonic()+min(.4,budget) if seconds else monotonic()+.4)
                    idle.clear()
            else:idle.discard(ix)
    results=[dict(s['result'],new_pages=s['pages'],work=dict(s['work']),range_finished=s['cursor']>=stop) for s in states]
    return results,dict(group_turns=[s['turns'] for s in states],first_turns=trace,improvements=improvements)
