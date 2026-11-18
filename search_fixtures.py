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


def signature_pairs(data):
    """The multiset of (side signature, matching signature) pairs a tile set can put on seams."""
    from collections import Counter
    sides=[]
    for tile in data['tiles']:
        row=[set(),set(),set()]
        for segment in tile['segments']:
            for side,point in segment:row[side].add(point)
        sides.extend(tuple(sorted(points)) for points in row)
    counts=Counter(sides);seen=set();pairs=[]
    for signature in sorted(counts):
        if signature in seen:continue
        mate=tuple(sorted(12-point for point in signature));seen.update((signature,mate))
        if counts[signature]!=counts[mate] or (signature==mate and counts[signature]%2):
            raise ValueError('The side signatures of this tile set do not pair up.')
        pairs.extend([(signature,mate)]*(counts[signature] if signature!=mate else counts[signature]//2))
    return pairs


def planted_inventory(source, seed=17, size=4):
    """A solvable edge-matching board with exactly the side-signature inventory of `source`.

    Every seam receives one (signature, matching signature) pair drawn without
    replacement from the pairs of the source tile set, so an exact engine sees
    the same supply of side signatures as on the source puzzle, but a full
    240-seam board is known to exist. Endpoints are joined into segments at
    random: the fixture is a control for seam matching only, not for the loop.
    Returns (data, board, reference codes).
    """
    from collections import Counter
    board=build_board(size);n=len(board.cells);rng=random.Random(seed)
    pairs=signature_pairs(source);edges=list(board.edges)
    if len(source['tiles'])!=n or len(pairs)!=len(edges):
        raise ValueError('The source must have one tile per cell of the board.')
    # A tile has an even number of endpoints, so the seams with an odd-sized
    # signature form closed curves between cells. On the sphere those are the
    # seams separating some set of mesh vertices from the rest.
    odd=[pair for pair in pairs if len(pair[0])%2];even=[pair for pair in pairs if len(pair[0])%2==0]
    ends=[tuple(set(board.cells[a].vertices)&set(board.cells[b].vertices)) for a,sa,b,sb in edges]
    count=1+max(vertex for pair in ends for vertex in pair)
    inside=[bool(odd) and rng.random()<0.5 for _ in range(count)]
    touching=[[index for index,pair in enumerate(ends) if vertex in pair] for vertex in range(count)]
    current=sum(inside[u]!=inside[w] for u,w in ends)
    for _ in range(1000000):
        if current==len(odd):break
        vertex=rng.randrange(count)
        delta=sum(1 if inside[ends[index][0]]==inside[ends[index][1]] else -1 for index in touching[vertex])
        if abs(current+delta-len(odd))<=abs(current-len(odd)) or rng.random()<0.05:
            inside[vertex]=not inside[vertex];current+=delta
    else:
        raise ValueError('No placement of the odd-sized signatures was found for this seed.')
    rng.shuffle(odd);rng.shuffle(even)
    chosen=[odd.pop() if inside[u]!=inside[w] else even.pop() for u,w in ends]
    # No tile may be empty: trade a blank seam of an all-blank cell for a marked even seam elsewhere.
    around=[[] for _ in range(n)]
    for index,(a,sa,b,sb) in enumerate(edges):around[a].append(index);around[b].append(index)
    marked=[index for index,(u,w) in enumerate(ends) if inside[u]==inside[w]]
    for _ in range(100000):
        empty=[cell for cell in range(n) if not any(chosen[index][0] for index in around[cell])]
        if not empty:break
        here=rng.choice(around[rng.choice(empty)]);there=rng.choice(marked)
        if chosen[there][0]:chosen[here],chosen[there]=chosen[there],chosen[here]
    else:
        raise ValueError('No placement without an empty tile was found for this seed.')
    sides=[[None]*3 for _ in range(n)]
    for (a,sa,b,sb),(signature,mate) in zip(edges,chosen):
        if rng.random()<0.5:signature,mate=mate,signature
        sides[a][sa]=signature;sides[b][sb]=mate
    order=list(range(n));rng.shuffle(order);tiles=[None]*n;reference=[None]*n
    for cell in range(n):
        points=[[side,point] for side in range(3) for point in sides[cell][side]];rng.shuffle(points)
        segments=[[points[index],points[index+1]] for index in range(0,len(points),2)]
        shift=rng.randrange(3);tile=order[cell]
        tiles[tile]={'id':f'planted-{tile}','segments':[[[(side+shift)%3,point] for side,point in segment] for segment in segments]}
        reference[cell]=3*tile+(-shift)%3
    data={'tiles':tiles,'fixture':'planted board with the side-signature inventory of a source tile set','seed':seed,'size':size}
    report=validate_arrangement(data,reference,board)
    if report['matched_edges']!=len(edges) or not report['unique_tiles']:raise AssertionError(report)
    if Counter(signature_pairs(data))!=Counter(pairs):raise AssertionError('inventory changed')
    return data,board,reference
