"""Deterministic disk-arc fixtures on the actual bipyramid, including 160-cell residuals."""
from random import Random

from geometry import build_board
from search_problem import SearchProblem
from r27_structure_filters import bits


def planted(subdivision=1, free=4, seed=23, split=False, uniform_ports=False):
    """Walk around a thickened spanning tree. A split replaces one turn by two loops."""
    board=build_board(subdivision)
    n=len(board.cells)
    rng=Random(seed)
    ports=[[] for _ in range(n)]
    seen={0}
    stack=[0]
    while stack:
        c=stack.pop()
        order=list(enumerate(board.neighbors[c]))
        rng.shuffle(order)
        for s,(b,t) in order:
            if b in seen:continue
            seen.add(b)
            stack.append(b)
            k=3 if uniform_ports else rng.randint(1,5)
            ports[c].extend([(s,k),(s,12-k)])
            ports[b].extend([(t,k),(t,12-k)])
    tiles=[]
    changed=False
    for c,items in enumerate(ports):
        items.sort()
        paths=[(items[j],items[(j+1)%len(items)]) for j in range(1,len(items),2)]
        if split and not changed and len(paths)>=2:
            paths=[(items[j],items[j+1]) for j in range(0,len(items),2)]
            changed=True
        tiles.append({'id':c,'segments':paths})
    p=SearchProblem(tiles,board)
    movable=sorted(rng.sample(range(n),min(free,n)))
    movable_mask=sum(7<<(3*c) for c in movable)
    domains=tuple(movable_mask if c in movable else 1<<(3*c) for c in range(n))
    return p,domains


def nested_loops():
    """Five nested disjoint latitudes on the actual triangular bipyramid."""
    board=build_board(1)
    data=[]
    for c in range(10):
        # Side 0 starts at the apex and side 2 ends there. Equal distance
        # from the apex is measured in opposite directions on these sides.
        levels=(2,4,6) if c<5 else (3,5)
        paths=[((0,k),(2,12-k)) for k in levels]
        data.append({'id':c,'segments':paths})
    p=SearchProblem(data,board)
    return p,tuple(1<<(3*c) for c in range(10))


def independent(problem,domains):
    """Independent permutation recursion with uniqueness only; final validation owns semantics."""
    order=sorted(range(problem.n),key=lambda c:domains[c].bit_count())
    answer=set()
    layout=[-1]*problem.n
    def visit(i,used):
        if i==len(order):
            if problem.validate(layout)['valid']:answer.add(tuple(layout))
            return
        c=order[i]
        for code in bits(domains[c]):
            bit=1<<(code//3)
            if used&bit:continue
            layout[c]=code
            visit(i+1,used|bit)
    visit(0,0)
    return answer

