"""Lower bounds on distinct retained physical reference bonds across alternative poses."""
from collections import defaultdict

from r27_structure_filters import bits


def pose_charges(optional, cells=None):
    charges = defaultdict(dict)
    direct = defaultdict(dict)
    for choices in optional.values():
        poses = set(choices) if cells is None else {(c,v) for c,v in choices if c in cells}
        if not poses:
            continue
        # A chosen board can entail this pair at both physical endpoints.
        # Charge one per endpoint, or two if only one endpoint has any mark.
        weight = 2 if len({code//3 for _,code in poses}) == 1 else 1
        for cell,code in poses:
            charges[cell][code] = charges[cell].get(code,0)+weight
            direct[cell][code] = direct[cell].get(code,0)+1
    return charges,direct


def projected_dual(domains, charges):
    from constraint_connectivity_round26 import assignment_upper_bound
    rows = {}
    zero = {}
    for c,values in charges.items():
        costs = {}
        for code in bits(domains[c]):
            t = code//3
            price = values.get(code,0)
            if t not in costs or price < costs[t]:
                costs[t] = price
        if costs:
            rows[c] = costs
            zero[c] = sum(v==0 for v in costs.values())
    # A cell with at least as many zero-price tiles as remaining cells can
    # always extend any assignment of the others at zero cost.
    active = sorted(rows,key=lambda c:(zero[c],c))
    while active and zero[active[-1]] >= len(active):
        active.pop()
    if not active or len(active)>32:
        return None
    tiles = sorted({t for c in active for t in rows[c]})
    if len(tiles)<len(active):
        return {'infeasible':True,'cells':active}
    high = max(max(rows[c].values()) for c in active)
    weights = [[high-rows[c][t] if t in rows[c] else None for t in tiles] for c in active]
    # Its integer dual is verified against every allowed cell/tile edge.
    cert = assignment_upper_bound(weights)
    return dict(cells=active,lower=len(active)*high-cert['bound'],high=high,
                row_price=dict(zip(active,cert['rows'])),column_price=dict(zip(tiles,cert['columns'])),
                checks=cert['dual_checks'],infeasible=False)


def cheapest_conflict(domains, charges, cells):
    """Rows reachable from unmatched rows in a maximum zero-cost matching."""
    zero={c:{code//3 for code in bits(domains[c]) if not charges[c].get(code,0)} for c in cells}
    mate={}
    def augment(c,seen):
        for t in sorted(zero[c]):
            if t in seen:continue
            seen.add(t)
            if t not in mate or augment(mate[t],seen):
                mate[t]=c
                return True
        return False
    for c in sorted(zero,key=lambda c:(len(zero[c]),c)):
        augment(c,set())
    region=set(cells)-set(mate.values())
    queue=list(region)
    while queue:
        for t in zero[queue.pop()]:
            other=mate.get(t)
            if other is not None and other not in region:
                region.add(other)
                queue.append(other)
    return region

def prune_alternatives(domains, optional, slack):
    result=dict(extra_minimum=0,unit_poses=0,over=False,assignment_rows=0,dual_checks=0,regional_reprices=0)
    if sum(bool(row) for row in optional.values()) <= slack:
        return result
    charges,direct=pose_charges(optional)
    cell_min = {c:min(values.values()) for c,values in charges.items() if len(values)==domains[c].bit_count()}
    by_tile=defaultdict(list)
    for entries in charges.values():
        for code,weight in entries.items():by_tile[code//3].append(weight)
    tile_min={t:min(row) for t,row in by_tile.items()
              if len(row)==sum(((mask>>(3*t)) & 7).bit_count() for mask in domains)}
    cell_total,tile_total=sum(cell_min.values()),sum(tile_min.values())
    threshold=2*slack
    lower=max(cell_total,tile_total)
    result['extra_minimum']=(lower+1)//2
    if lower>threshold:
        result['over']=True
        return result
    regional=None
    regional_prices={}
    dual=projected_dual(domains,charges)
    if dual:
        result['assignment_rows']=len(dual['cells'])
        if dual['infeasible']:
            result.update(over=True,assignment_infeasible=True)
            return result
        result['dual_checks']=dual['checks']
        result['extra_minimum']=(max(lower,dual['lower'])+1)//2
        if dual['lower']>threshold:
            result['over']=True
            return result
        if 0<dual['lower']<=threshold:
            # Price only these rows. A bond with one physical endpoint in
            # this region can contribute its full two units there, while its
            # other endpoint is uncharged in this separate certificate.
            region=cheapest_conflict(domains,charges,dual['cells'])
            regional_prices,_=pose_charges(optional,region)
            if any(w>charges[c][code] for c,row in regional_prices.items() for code,w in row.items()):
                regional=projected_dual(domains,regional_prices)
                if regional and regional['infeasible']:
                    result.update(over=True,assignment_infeasible=True)
                    return result
                if regional:
                    result['regional_reprices']=1
                    result['dual_checks']+=regional['checks']
                    result['extra_minimum']=max(result['extra_minimum'],(regional['lower']+1)//2)
                    if regional['lower']>threshold:
                        result['over']=True
                        return result
    active=dual['row_price'] if dual else {}
    columns=dual['column_price'] if dual else {}
    # An outside cell using this tile prevents the priced rows from using its
    # column; the unused nonnegative column potential strengthens their bound.
    forbidden=sum(7 << (3*t) for t,v in columns.items() if dual['lower']+v>threshold)
    region_cells=regional['row_price'] if regional else {}
    region_cols=regional['column_price'] if regional else {}
    region_forbidden=sum(7 << (3*t) for t,v in region_cols.items() if regional['lower']+v>threshold)
    for c,mask in enumerate(domains):
        remove=(mask & forbidden) if c not in active else 0
        if c not in region_cells:
            remove |= mask & region_forbidden
        values=charges.get(c,{})
        for code in bits(mask) if c in active or c in region_cells else values:
            t=code//3
            w=values.get(code,0)
            cost=max(cell_total-cell_min.get(c,0)+w,tile_total-tile_min.get(t,0)+w)
            if dual:
                cost=max(cost,dual['lower']+columns.get(t,0)+w+(active[c]-dual['high'] if c in active else 0))
            if regional:
                price=regional_prices.get(c,{}).get(code,0)
                reduced=region_cells[c]-regional['high'] if c in region_cells else 0
                cost=max(cost,regional['lower']+region_cols.get(t,0)+price+reduced)
            if cost>threshold or direct.get(c,{}).get(code,0)>slack:
                remove |= 1<<code
        result['unit_poses'] += (mask & remove).bit_count()
        domains[c] &= ~remove
        if not domains[c]:result['over']=True
    return result
