"""Positive 365-segment control with the original board geometry and synthetic tiles."""
from collections import Counter,defaultdict
from random import Random

from geometry import build_board
from search_problem import SearchProblem


def planted_365():
    b=build_board(4)
    n=len(b.cells)
    incident=defaultdict(list)
    for c in b.cells:
        for corner in c.vertices:incident[corner].append(c.index)
    apex=next(set(cells) for cells in incident.values() if len(cells)==5)
    multiplicity=Counter()
    adjacent=[[] for _ in b.cells]
    for e,(a,s,c,t) in enumerate(b.edges):
        adjacent[a].append((e,c));adjacent[c].append((e,a))
        if a in apex and c in apex:multiplicity[e]+=1
    seen={0};todo=[0]
    while todo:
        a=todo.pop()
        for e,c in adjacent[a]:
            if c in seen:continue
            seen.add(c);todo.append(c);multiplicity[e]+=2
    extra=[i for i in range(len(b.edges)) if multiplicity[i]<2][:21]
    for e in extra:multiplicity[e]+=2
    rng=Random(365)
    half=[[] for _ in b.cells];point={}
    index=0
    for e,(a,s,c,t) in enumerate(b.edges):
        for k in rng.sample(range(1,12),multiplicity[e]):
            half[a].append((index,c));half[c].append((index,a))
            point[a,index]=(s,k);point[c,index]=(t,12-k)
            index+=1
    if index!=365:raise AssertionError('Expected 365 seam links.')
    used=set();nodes=[0];incoming=[];path=[]
    while nodes:
        a=nodes[-1]
        while half[a] and half[a][-1][0] in used:half[a].pop()
        if half[a]:
            e,c=half[a].pop();used.add(e)
            nodes.append(c);incoming.append((a,c,e))
        else:
            nodes.pop()
            if incoming:path.append(incoming.pop())
    path.reverse()
    tiles=[{'id':c,'segments':[]} for c in range(n)]
    for i,(a,c,e) in enumerate(path):
        d,other,following=path[(i+1)%len(path)]
        if d!=c:raise AssertionError('Euler order incomplete.')
        tiles[c]['segments'].append((point[c,e],point[c,following]))
    p=SearchProblem(tiles,b)
    fixed=tuple(1<<(3*c) for c in range(n))
    if not p.validate([3*c for c in range(n)])['valid']:raise AssertionError('Invalid synthetic 365-strand witness.')
    return p,fixed

