"""Independent finite pose packing certificate from fixed physical side coincidences."""
from functools import lru_cache


def check_packing(problem,bonds,max_steps=100000):
    incidence={}
    for a,sa,b,sb in problem.edges:
        incidence[a,sa]=(b,sb)
        incidence[b,sb]=(a,sa)
    graph={}
    for a,b in bonds:
        for left,right in ((a,b),(b,a)):
            tile,side=divmod(left,3);other,back=divmod(right,3)
            graph.setdefault(tile,[]).append((side,other,back))
    unreached=set(graph)
    pieces=[]
    while unreached:
        root=min(unreached)
        component={root}; stack=[root]
        while stack:
            for _,other,_ in graph[stack.pop()]:
                if other not in component:
                    component.add(other);stack.append(other)
        unreached.difference_update(component)
        masks=set()
        for cell in range(problem.n):
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
                    masks.add(occupied)
        pieces.append(tuple(sorted(masks)))
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
