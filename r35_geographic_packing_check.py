"""Independent finite pose packing certificate from fixed physical side coincidences."""
from functools import lru_cache


def component_poses(n,incidence,bonds):
    """Yield each labelled embedding of one connected set under every root pose."""
    graph={}
    for a,b in bonds:
        for left,right in ((a,b),(b,a)):
            tile,side=divmod(left,3);other,back=divmod(right,3)
            graph.setdefault(tile,[]).append((side,other,back))
    root=min(graph)
    for cell in range(n):
        for turn in range(3):
            assigned={root:(cell,turn)}
            occupied=1<<cell
            pending=[root]
            while pending:
                tile=pending.pop();at,rotation=assigned[tile]
                for side,other,back in graph[tile]:
                    edge=incidence.get((at,(side+rotation)%3))
                    if edge is None:break
                    dest,facing=edge;pose=(dest,(facing-back)%3)
                    if other in assigned:
                        if pose!=assigned[other]:break
                    elif occupied>>dest&1:
                        break
                    else:
                        assigned[other]=pose;occupied|=1<<dest;pending.append(other)
                else:
                    continue
                break
            else:
                if len(assigned)!=len(graph):
                    raise ValueError('Expected one connected component.')
                yield occupied, assigned


def component_masks(n,incidence,bonds):
    return tuple(sorted({mask for mask,_ in component_poses(n,incidence,bonds)}))


def check_packing(problem,bonds,max_steps=100000):
    bonds=frozenset(bonds)
    incidence={}
    for a,sa,b,sb in problem.edges:
        incidence[a,sa]=(b,sb)
        incidence[b,sb]=(a,sa)
    neighbors={}
    for a,b in bonds:
        if not 0<=a<b<3*problem.n or a//3==b//3:
            raise ValueError('Expected a physical side pair on distinct tiles.')
        x,y=a//3,b//3
        neighbors.setdefault(x,set()).add(y);neighbors.setdefault(y,set()).add(x)
    unreached=set(neighbors)
    pieces=[]
    while unreached:
        root=min(unreached);component={root};stack=[root]
        while stack:
            for other in neighbors[stack.pop()]-component:
                component.add(other);stack.append(other)
        unreached.difference_update(component)
        local=[(a,b) for a,b in bonds if a//3 in component]
        pieces.append(component_masks(problem.n,incidence,local))
    pieces.sort(key=len)
    steps=0
    class Limit(Exception):pass
    @lru_cache(maxsize=65536)
    def fits(i,used):
        nonlocal steps
        steps+=1
        if steps>max_steps:raise Limit
        if i==len(pieces):return True
        return any(not used&mask and fits(i+1,used|mask) for mask in pieces[i])
    try:feasible=fits(0,0)
    except Limit:feasible=None
    return {'feasible':feasible,'poses_per_component':[len(x) for x in pieces],'nodes':steps}
