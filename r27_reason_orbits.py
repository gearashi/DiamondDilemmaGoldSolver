"""Exact same-residual choice stabilizer for Gold existence searches."""
from collections import Counter
from copy import copy
from constraint_region_isomorphism import residual_graph, verified_isomorphism, _Work, IsomorphismLimit
from constraint_regions import HierarchicalRegionPruner
from r27_reason_maxsat import domain_rows


def orbit_choices(problem, *, domains=None, cell=None, work_limit=60000, max_vertices=400, max_active=8, check=lambda:None):
    rows = domain_rows(problem, domains)
    active = [i for i,row in enumerate(rows) if len(row)>1]
    if cell is None:
        cell = min(active,key=lambda c:(len(rows[c]),c)) if active else 0
    if not 0 <= cell < problem.n:
        raise ValueError('Cell is outside the board.')
    result = {'cell':cell,'choices':list(rows[cell]),'suppressed':{},'status':'skipped','stats':{}}
    if not all(rows) or len(active)>max_active or cell not in active:
        return result
    stats = Counter()
    work = _Work(work_limit,check,stats)
    try:
        domain_masks = [sum(1<<code for code in row) for row in rows]
        graph = residual_graph(HierarchicalRegionPruner(problem),domain_masks,work,max_vertices)
        if graph is None:
            return result
        pins = {code:node for node,(c,code) in graph.choices.items() if c==cell}
        representatives = []
        for value,node in pins.items():
            for representative in representatives:
                left,right = copy(graph),copy(graph)
                left.colors,right.colors = list(graph.colors),list(graph.colors)
                left.colors[node] = right.colors[pins[representative]] = max(graph.colors)+1
                witness = verified_isomorphism(left,right,work,stats,4000)
                if witness is not None:
                    result['suppressed'][value] = representative
                    break
            else:
                representatives.append(value)
        result['status'] = 'completed'
    except IsomorphismLimit:
        result['status'] = 'bounded'
    finally:
        # Only fully replayed mappings suppress alternatives. Values omitted by
        # arc consistency are left to the caller's ordinary propagator.
        result['choices'] = [v for v in rows[cell] if v not in result['suppressed']]
        result['stats'] = dict(stats)
    return result

