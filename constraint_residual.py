"""Exact residual Gold circuits after contracting forced degree-two paths.

A fixed physical tile has one local segment at every occupied port. Across a
fixed/fixed seam its ports also have exactly one seam edge. Such components are
paths between distinct *port identities* of unfinished cells, or closed loops.
Replacing each path by one mandatory edge preserves every possible Gold loop.
Crossings never become vertices; parallel physical connections stay distinct.
"""
from collections import defaultdict
from functools import lru_cache

from constraint_circuit import CircuitModel, CircuitLimit


@lru_cache(maxsize=8)
def _rotation_tables(geometry):
    """Immutable exact-key cache; changed tile geometry cannot reuse old data."""
    pairs, masks = [], []
    for segments in geometry:
        for rotation in range(3):
            row = tuple((((a[0]+rotation)%3)*11+a[1]-1,
                         ((b[0]+rotation)%3)*11+b[1]-1) for a,b in segments)
            sides = [0, 0, 0]
            for segment in row:
                for port in segment: sides[port//11] |= 1 << (port%11)
            pairs.append(row); masks.append(tuple(sides))
    return tuple(pairs), tuple(masks)


def _reverse(value):
    return int(f'{value:011b}'[::-1], 2)


def _narrow_fixed(problem, rows, masks, check, stats):
    """Only forced inventory and forced seam deductions; no search decisions."""
    rows = [tuple(row) for row in rows]
    adjacent = [[] for _ in rows]
    for a,sa,b,sb in problem.edges:
        adjacent[a].append((sa,b,sb)); adjacent[b].append((sb,a,sa))
    while True:
        check()
        if any(not row for row in rows): return rows, 'empty_domain'
        fixed = {cell:row[0] for cell,row in enumerate(rows) if len(row)==1}
        used = {code//3 for code in fixed.values()}
        if len(used)!=len(fixed): return rows, 'duplicate_fixed_tile'
        changed = False
        for cell,row in enumerate(rows):
            check()
            # A fixed cell keeps its own tile. Every other cell must relinquish
            # all three rotations of already occupied physical pieces.
            inventory = row if cell in fixed else tuple(code for code in row if code//3 not in used)
            stats['circuit_fixed_inventory_removals'] += len(row)-len(inventory)
            required = [(side,_reverse(masks[fixed[other]][other_side]))
                        for side,other,other_side in adjacent[cell] if other in fixed]
            required += [(side,0) for side in range(3) if int(problem.neighbors[cell,side])<0]
            narrowed = tuple(code for code in inventory if all(masks[code][side]==value for side,value in required))
            stats['circuit_fixed_seam_removals'] += len(inventory)-len(narrowed)
            if narrowed!=row: rows[cell]=narrowed; changed=True
        if not changed: return rows, None


def _fixed_paths(problem, rows, pairs, check):
    """Return (paths, fixed ports, cycles, error); each path retains its anchor."""
    fixed = {cell:row[0] for cell,row in enumerate(rows) if len(row)==1}
    local, seam, external = {}, {}, {}
    anchor = None
    for cell,code in fixed.items():
        check()
        for a,b in pairs[code]:
            left,right=(cell,a),(cell,b)
            local[left]=right;local[right]=left
        if code//3==0:
            a,b=pairs[code][0];anchor=((cell,a),(cell,b))
    # Read exact board edges, preserving port p -> 10-p across each seam.
    side_map = {}
    for a,sa,b,sb in problem.edges:
        side_map[a,sa]=(b,sb);side_map[b,sb]=(a,sa)
    for port in local:
        check();cell,p=port
        other = side_map.get((cell,p//11))
        if other is None:return [],len(local),0,'fixed_boundary_port'
        target=(other[0],11*other[1]+10-p%11)
        if other[0] in fixed:
            if target not in local:return [],len(local),0,'fixed_seam_mismatch'
            seam[port]=target
        else:
            seam[port]=target
            # A board side has exactly one neighbor, so open port identity is
            # unique even when several seams connect the same pair of cells.
            if target in external:return [],len(local),0,'duplicate_terminal'
            external[target]=port
    visited,paths=set(),[]
    for start in sorted(external):
        check()
        if external[start] in visited:continue
        current=external[start];direction=None
        while True:
            check();visited.add(current)
            mate=local[current];visited.add(mate)
            if anchor==(current,mate):direction=True
            elif anchor==(mate,current):direction=False
            end=seam[mate]
            if end not in local:
                paths.append((start,end,direction));break
            current=end
    cycles=0
    for start in local:
        if start in visited:continue
        cycles+=1;current=start
        while current not in visited:
            check();visited.add(current);mate=local[current];visited.add(mate)
            current=seam[mate]
    return paths,len(local),cycles,None


def build_residual_circuit(problem, domains, *, check, group_supports,
                           break_direction, max_arcs, max_link_terms):
    """Build from already validated domains; fixed code variables stay named."""
    from ortools.sat.python import cp_model
    check()
    geometry=tuple(tuple(tuple(tuple(point) for point in segment) for segment in tile['segments'])
                   for tile in problem.tiles)
    pairs,masks=_rotation_tables(geometry)
    stats={'connectivity_encoding':'contracted_fixed_endpoint_circuit',
           'circuit_nodes':0,'circuit_arcs':0,'circuit_segment_edges':0,
           'circuit_seam_edges':0,'circuit_placements':sum(map(len,domains)),
           'circuit_support_groups':0,'circuit_reused_groups':0,
           'circuit_complement_groups':0,'circuit_link_terms':0,
           'circuit_direction_symmetry':bool(break_direction),
           'circuit_fixed_inventory_removals':0,'circuit_fixed_seam_removals':0,
           'circuit_fixed_cells':0,'circuit_contracted_ports':0,
           'circuit_contracted_paths':0,'circuit_fixed_cycles':0,
           'circuit_residual_placements':0}
    rows,error=_narrow_fixed(problem,domains,masks,check,stats)
    fixed={cell:row[0] for cell,row in enumerate(rows) if len(row)==1}
    used={code//3 for code in fixed.values()}
    stats['circuit_fixed_cells']=len(fixed)
    stats['circuit_residual_placements']=sum(len(row) for row in rows if len(row)>1)
    paths=[]
    if error is None:
        paths,ports,cycles,error=_fixed_paths(problem,rows,pairs,check)
        stats.update(circuit_contracted_ports=ports,circuit_contracted_paths=len(paths),
                     circuit_fixed_cycles=cycles)
        if cycles and (len(fixed)!=problem.n or cycles!=1):error='closed_proper_fixed_loop'
        if not cycles and len(fixed)==problem.n:error='no_complete_fixed_loop'
    model=cp_model.CpModel();codes=[];placements=[];by_tile=[[] for _ in rows]
    for cell,row in enumerate(rows):
        check()
        code_var=model.new_int_var_from_domain(cp_model.Domain.from_values(row or (0,)),f'code_{cell}')
        codes.append(code_var)
        if cell in fixed:
            placements.append({fixed[cell]:model.new_constant(1)})
        else:
            literals={code:model.new_bool_var(f'p_{cell}_{code}') for code in row}
            placements.append(literals);model.add_exactly_one(list(literals.values()))
            model.add(code_var==sum(code*literal for code,literal in literals.items()))
            for code,literal in literals.items():by_tile[code//3].append(literal)
    for tile,literals in enumerate(by_tile):
        check()
        if tile not in used:model.add_exactly_one(literals)
    def finish():
        check();stats.update(circuit_variables=len(model.proto.variables),
                             circuit_constraints=len(model.proto.constraints))
        return CircuitModel(model,codes,placements,stats)
    if error:
        model.add_bool_or([]);stats['circuit_residual_rejection']=error
        return finish()
    # The only forced component is the whole valid board: no optional circuit
    # is needed. Named fixed code variables reconstruct the full certificate.
    if len(fixed)==problem.n:return finish()
    arcs=[];port_nodes={};port_active={};directed_local={}
    def append_arc(source,destination,literal):
        if len(arcs)>=max_arcs:raise CircuitLimit(f'Direct circuit arc limit exceeded ({max_arcs}).')
        arcs.append((source,destination,literal))
    for cell,row in enumerate(rows):
        if cell in fixed:continue
        check();edges=defaultdict(list);ports=defaultdict(list)
        for offset,code in enumerate(row):
            if offset%32==0:check()
            for a,b in pairs[code]:
                edges[tuple(sorted((a,b)))].append(code)
                ports[a].append(code);ports[b].append(code)
        all_codes=frozenset(row);groups={}
        def support(selected):
            key=frozenset(selected)
            if not key:return model.new_constant(0)
            if len(key)==len(row):return model.new_constant(1)
            if len(key)==1:return placements[cell][next(iter(key))]
            if group_supports and key in groups:
                stats['circuit_reused_groups']+=1;return groups[key]
            complement=all_codes-key
            if group_supports and complement in groups:
                stats['circuit_complement_groups']+=1;return groups[complement].Not()
            stats['circuit_link_terms']+=len(key)
            if stats['circuit_link_terms']>max_link_terms:
                raise CircuitLimit(f'Direct circuit linking-term limit exceeded ({max_link_terms}).')
            value=model.new_bool_var(f'present_{cell}_{stats["circuit_support_groups"]}')
            model.add(sum(placements[cell][code] for code in sorted(key))==value)
            stats['circuit_support_groups']+=1
            if group_supports:groups[key]=value
            return value
        for port,selected in sorted(ports.items()):
            check();node=len(port_nodes);port_nodes[cell,port]=node
            active=support(selected);port_active[cell,port]=active
            append_arc(node,node,active.Not())
        for offset,((a,b),selected) in enumerate(sorted(edges.items())):
            if offset%32==0:check()
            present=support(selected)
            forward=model.new_bool_var(f'line_{cell}_{a}_{b}_f')
            backward=model.new_bool_var(f'line_{cell}_{a}_{b}_b')
            model.add(forward+backward==present)
            append_arc(port_nodes[cell,a],port_nodes[cell,b],forward)
            append_arc(port_nodes[cell,b],port_nodes[cell,a],backward)
            directed_local[cell,a,b]=forward;directed_local[cell,b,a]=backward
            stats['circuit_segment_edges']+=1
        if break_direction:
            for code in row:
                if code//3==0:
                    a,b=pairs[code][0]
                    model.add_implication(placements[cell][code],directed_local[cell,a,b])
    for a,sa,b,sb in problem.edges:
        if a in fixed or b in fixed:continue
        check()
        for p in range(11):
            left,right=(a,11*sa+p),(b,11*sb+10-p)
            if left not in port_nodes and right not in port_nodes:continue
            if left not in port_nodes:model.add(port_active[right]==0);continue
            if right not in port_nodes:model.add(port_active[left]==0);continue
            forward=model.new_bool_var(f'seam_{a}_{sa}_{p}_f')
            backward=model.new_bool_var(f'seam_{a}_{sa}_{p}_b')
            model.add(forward+backward==port_active[left])
            model.add(port_active[left]==port_active[right])
            append_arc(port_nodes[left],port_nodes[right],forward)
            append_arc(port_nodes[right],port_nodes[left],backward)
            stats['circuit_seam_edges']+=1
    for index,(left,right,direction) in enumerate(paths):
        check()
        if left not in port_nodes or right not in port_nodes:
            model.add_bool_or([]);continue
        # Each path is mandatory and indivisible; both endpoints must be used.
        model.add(port_active[left]==1);model.add(port_active[right]==1)
        forward=model.new_bool_var(f'fixed_path_{index}_f')
        backward=model.new_bool_var(f'fixed_path_{index}_b')
        model.add(forward+backward==1)
        append_arc(port_nodes[left],port_nodes[right],forward)
        append_arc(port_nodes[right],port_nodes[left],backward)
        if break_direction and direction is not None:
            model.add((forward if direction else backward)==1)
    check()
    if arcs:model.add_circuit(arcs)
    else:model.add_bool_or([])
    stats.update(circuit_nodes=len(port_nodes),circuit_arcs=len(arcs))
    return finish()
