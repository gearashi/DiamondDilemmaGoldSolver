"""Reproducible connected-loop fixtures; no third-party puzzle diagrams."""
from __future__ import annotations
import random
from geometry import build_board
from validator import validate_arrangement

def planted_gold(seed=17, size=4, extra_edges=24):
    board=build_board(size)
    n=len(board.cells)
    rng=random.Random(seed)
    neighbors=board.neighbor_cells
    # A doubled spanning tree is connected and even-degree. Extra doubled
    # edges add variety without changing Eulerian connectivity.
    seen={0}; stack=[0]; selected=set()
    while stack:
        a=stack[-1]
        options=[int(b) for b in neighbors[a] if int(b) not in seen]
        if not options: stack.pop(); continue
        b=rng.choice(options); seen.add(b);stack.append(b);selected.add(tuple(sorted((a,b))))
    remaining=[(a,b) for a,sa,b,sb in board.edges if tuple(sorted((a,b))) not in selected]
    rng.shuffle(remaining)
    selected.update(tuple(sorted(pair)) for pair in remaining[:extra_edges])
    links=[]; adjacency=[[] for _ in range(n)]
    for a,sa,b,sb in board.edges:
        if tuple(sorted((a,b))) not in selected: continue
        for p in rng.sample(range(1,12),2):
            index=len(links);links.append(((a,sa,p),(b,sb,12-p)))
            adjacency[a].append((index,b));adjacency[b].append((index,a))
    for values in adjacency:rng.shuffle(values)
    used=set();nodes=[0];incoming=[];circuit=[]
    while nodes:
        a=nodes[-1]
        while adjacency[a] and adjacency[a][-1][0] in used:adjacency[a].pop()
        if adjacency[a]:
            edge,b=adjacency[a].pop();used.add(edge);nodes.append(b);incoming.append((a,b,edge))
        else:
            nodes.pop()
            if incoming:circuit.append(incoming.pop())
    circuit.reverse()
    assert len(circuit)==len(links)
    originals=[{'id':f'synthetic-{i}','segments':[]} for i in range(n)]
    def point(edge,cell):
        return next([side,p] for c,side,p in links[edge] if c==cell)
    for index,(a,b,edge) in enumerate(circuit):
        outgoing=circuit[(index+1)%len(circuit)]
        assert outgoing[0]==b
        originals[b]['segments'].append([point(edge,b),point(outgoing[2],b)])
    permutation=list(range(n));rng.shuffle(permutation)
    tiles=[];reference=[None]*n
    for index,cell in enumerate(permutation):
        shift=rng.randrange(3)
        tile=originals[cell]
        tiles.append({'id':tile['id'],'segments':[[[(s+shift)%3,p] for s,p in segment] for segment in tile['segments']]})
        reference[cell]=3*index+(-shift)%3
    data={'tiles':tiles,'fixture':'planted single Eulerian gold loop','seed':seed,'size':size}
    report=validate_arrangement(data,reference,board)
    if not report['valid']:raise AssertionError(report)
    return data,board,reference
