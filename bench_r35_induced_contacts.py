"""Check contacts forced by labelled rigid poses in physically embeddable departure sets."""
import argparse
from collections import deque
import json
from pathlib import Path

from bench_r35_geographic_components import parts
from bench_r35_loss_boundary344 import problem_scope
from r35_geographic_packing_check import component_poses


def incidence(problem):
    table={}
    for u,a,v,b in problem.edges:table[u,a]=(v,b);table[v,b]=(u,a)
    return table


def wrong_contacts(problem, placed):
    slots=getattr(problem,'_induced_raw_slots',None)
    if slots is None:
        slots=[]
        for tile in problem.tiles:
            row=[[] for _ in range(3)]
            for segment in tile['segments']:
                for s,pos in segment:row[s].append(pos)
            slots.append(tuple(tuple(sorted(side)) for side in row))
        problem._induced_raw_slots=slots
    codes={cell:3*t+rot for t,(cell,rot) in placed.items()}
    failures=[]
    for a,s,b,t in problem.edges:
        if a not in codes or b not in codes:continue
        u,v=codes[a],codes[b]
        left=slots[u//3][(s-u%3)%3];right=slots[v//3][(t-v%3)%3]
        if left!=tuple(12-pos for pos in reversed(right)):
            failures.append((u//3,v//3))
    return failures


def assess(problem, table, group):
    rows=[wrong_contacts(problem,assignment) for _,assignment in component_poses(problem.n,table,group)]
    return {'poses':len(rows),'compatible':sum(not r for r in rows),
            'min_mismatches':min(map(len,rows),default=None),'max_mismatches':max(map(len,rows),default=None)}


def path(edges, start, goal):
    graph={}
    for e in sorted(edges):
        u,v=e[0]//3,e[1]//3
        graph.setdefault(u,[]).append((v,e));graph.setdefault(v,[]).append((u,e))
    prev={start:None};queue=deque([start])
    while queue:
        u=queue.popleft()
        if u==goal:break
        for v,e in graph[u]:
            if v not in prev:prev[v]=(u,e);queue.append(v)
    result=[]
    while goal!=start:
        goal,e=prev[goal];result.append(e)
    return frozenset(result)


def screen(problem,ref,required,donors):
    table=incidence(problem)
    reports=[]
    allpaths=set()
    for name,losses in donors:
        chosen=(ref-losses)|required
        candidates=set()
        large=max(parts(chosen),key=len)
        for _,assigned in component_poses(problem.n,table,large):
            for u,v in wrong_contacts(problem,assigned):
                candidates.add(path(large,u,v))
        allpaths.update(candidates)
        reports.append(dict(name=name,retained=len(ref-losses),input_component=len(large),
                            geometry=assess(problem,table,large),paths=len(candidates)))
    verified=[]
    for group in sorted(allpaths,key=lambda r:(len(r),sorted(r))):
        if any(frozenset(map(tuple,old['core']))<=group for old in verified):continue
        outcome=assess(problem,table,group)
        if outcome['poses'] and not outcome['compatible']:
            verified.append(dict(core=sorted(group),**outcome))
    return dict(puzzle=problem.fingerprint,sources=reports,paths=len(allpaths),verified=verified)


if __name__=='__main__':
    import sys
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('--tiles',required=True)
    cli.add_argument('--evidence',required=True,type=Path);cli.add_argument('--output',required=True,type=Path)
    a=cli.parse_args();p,ref,_,req=problem_scope(a.tiles);d=json.loads(a.evidence.read_bytes())
    donors=[('supported-25',set(map(tuple,d['positive_final']['lost'])))]
    donors.extend((r['phase'],set(map(tuple,r['witness_departures']))) for r in d['discovery'] if r['witness_departures'])
    a.output.write_text(json.dumps(screen(p,ref,req,donors),indent=2)+'\n',encoding='utf8')
