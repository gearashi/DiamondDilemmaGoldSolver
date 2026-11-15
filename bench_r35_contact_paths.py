"""Enumerate material physical paths between mismatched incumbent-adjacent tiles."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from bench_r35_geographic_loss_cores import connected_pose_count
from bench_r35_induced_contacts import assess, incidence
from bench_r35_loss_boundary344 import problem_scope
from r35_induced_contacts import InducedContacts


def paths(graph,start,goal,limit):
    distance={goal:0};queue=[goal]
    for tile in queue:
        for v,_ in graph[tile]:
            if v not in distance:distance[v]=1+distance[tile];queue.append(v)
    def walk(u,seen,chosen):
        if u==goal:
            if len(chosen)==limit:yield frozenset(chosen)
            return
        if len(chosen)+distance.get(u,limit+1)>limit:return
        for v,pair in graph[u]:
            if v not in seen:yield from walk(v,seen|{v},chosen+[pair])
    yield from walk(start,{start},[])


def search(tiles, seeds, maximum, output, minimum=3):
    p,ref,_,req=problem_scope(tiles)
    incumbent=json.loads((Path(__file__).parent/'docs/R29_INCUMBENT.json').read_bytes())['codes']
    mismatches=p.validate(incumbent)['unmatched_edges']
    graph=defaultdict(list)
    for pair in sorted(ref):
        u,v=pair[0]//3,pair[1]//3
        graph[u].append((v,pair));graph[v].append((u,pair))
    checker=InducedContacts(p);table=incidence(p)
    initial=json.loads(seeds.read_bytes())
    known=[frozenset(map(tuple,x)) for x in initial['edge_perfect_cores']]
    layers=[]
    for length in range(minimum,maximum+1):
        counts=dict(length=length,paths=0,dominated=0,checked=0,new=0)
        for x in mismatches:
            start,end=incumbent[x['cell_a']]//3,incumbent[x['cell_b']]//3
            for g in paths(graph,start,end,length):
                counts['paths']+=1
                if any(k<=g for k in known):counts['dominated']+=1;continue
                counts['checked']+=1
                geom,ok=checker.count(g)
                if not geom:raise ArithmeticError('Incumbent-containing path lacks any geographic embedding.')
                if not ok:
                    independent=assess(p,table,g)
                    if independent['poses']!=geom or independent['compatible']:
                        raise ArithmeticError('Independent raw slots failed to confirm the pose conflict.')
                    known.append(g);counts['new']+=1
        layers.append(counts)
        report=dict(puzzle=p.fingerprint,scope='all simple incumbent paths of each exact tested length between its 33 mismatching contacts',
                    layers=layers,edge_perfect_cores=[sorted(g) for g in known])
        output.write_text(json.dumps(report,separators=(',',':'))+'\n',encoding='utf8')
        print(counts,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiles',required=True);parser.add_argument('--seed',required=True,type=Path)
    parser.add_argument('--length',type=int,default=13);parser.add_argument('--minimum',type=int,default=3)
    parser.add_argument('--output',required=True,type=Path)
    a=parser.parse_args();search(a.tiles,a.seed,a.length,a.output,a.minimum)
