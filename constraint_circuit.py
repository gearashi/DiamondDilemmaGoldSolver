"""Direct single-Gold-loop CP-SAT model over optional endpoint nodes.

A selected tile forces every one of its undirected segment edges into the
circuit in one direction. Occupied seam endpoints are connected across the
actual board edge, with reversed endpoint order. Inactive ports take a self
loop. AddCircuit then requires one cycle containing ALL active ports, not a
collection of separately closed components. Stroke crossings are not nodes.

Official API contract: https://raw.githubusercontent.com/google/or-tools/v9.15/ortools/sat/python/cp_model.py
Model construction is capped; exceeding a cap is unsupported/inconclusive,
never evidence that the encoded puzzle has no solution.
"""
from collections import defaultdict
from dataclasses import dataclass

MAX_CELLS=160
MAX_PLACEMENTS=80_000
MAX_CIRCUIT_ARCS=200_000
MAX_LINK_TERMS=1_000_000


class CircuitLimit(Exception):
    pass


@dataclass
class CircuitModel:
    model: object
    codes: list
    placements: list
    stats: dict


def build_circuit_model(problem,domains,*,check=lambda:None,group_supports=True,
                        break_direction=True,max_cells=MAX_CELLS,
                        max_placements=MAX_PLACEMENTS,max_arcs=MAX_CIRCUIT_ARCS,
                        max_link_terms=MAX_LINK_TERMS):
    from ortools.sat.python import cp_model
    check()
    if len(domains)!=problem.n:raise ValueError('One orientation domain is required per cell.')
    if problem.n>max_cells:raise CircuitLimit(f'Direct circuit model supports at most {max_cells} cells.')
    domains=[tuple(sorted(set(row))) for row in domains]
    if any(type(code)is not int or not 0<=code<3*problem.n for row in domains for code in row):
        raise ValueError('Circuit model has an invalid orientation code.')
    if any(not set(row).issubset(problem.domains[cell]) for cell,row in enumerate(domains)):
        raise ValueError('Circuit domains must be subsets of the problem domains.')
    placement_count=sum(map(len,domains))
    if placement_count>max_placements:
        raise CircuitLimit(f'Direct circuit placement limit exceeded ({placement_count}>{max_placements}).')
    model=cp_model.CpModel();codes=[];placements=[];by_tile=[[] for _ in range(problem.n)]
    stats={'connectivity_encoding':'direct_optional_endpoint_circuit','circuit_nodes':0,
           'circuit_arcs':0,'circuit_segment_edges':0,'circuit_seam_edges':0,
           'circuit_placements':placement_count,'circuit_support_groups':0,
           'circuit_reused_groups':0,'circuit_complement_groups':0,'circuit_link_terms':0,
           'circuit_direction_symmetry':bool(break_direction)}
    segment_pairs=[]
    for tile in problem.tiles:
        for rotation in range(3):
            segment_pairs.append(tuple((((a[0]+rotation)%3)*11+a[1]-1,
                                        ((b[0]+rotation)%3)*11+b[1]-1)
                                       for a,b in tile['segments']))
    for cell,row in enumerate(domains):
        check()
        literals={code:model.new_bool_var(f'p_{cell}_{code}') for code in row}
        placements.append(literals);model.add_exactly_one(list(literals.values()))
        code_var=model.new_int_var_from_domain(cp_model.Domain.from_values(row or (0,)),f'code_{cell}')
        model.add(code_var==sum(code*literal for code,literal in literals.items()))
        codes.append(code_var)
        for code,literal in literals.items():by_tile[code//3].append(literal)
    for literals in by_tile:
        check();model.add_exactly_one(literals)
    arcs=[];port_nodes={};port_active={};directed_local={}
    def append_arc(source,destination,literal):
        if len(arcs)>=max_arcs:raise CircuitLimit(f'Direct circuit arc limit exceeded ({max_arcs}).')
        arcs.append((source,destination,literal))
    for cell,row in enumerate(domains):
        check();edges=defaultdict(list);ports=defaultdict(list)
        for offset,code in enumerate(row):
            if offset%32==0:check()
            for a,b in segment_pairs[code]:
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
        # Every physical tile is used. Fix the traversal direction of tile0's
        # first segment conditionally on its placement; reversal of a Gold
        # loop always satisfies one of the two directions, so no board is lost.
        if break_direction:
            for code in row:
                if code//3==0:
                    a,b=segment_pairs[code][0]
                    model.add_implication(placements[cell][code],directed_local[cell,a,b])
        for port in ports:
            if int(problem.neighbors[cell,port//11])<0:model.add(port_active[cell,port]==0)
    for a,sa,b,sb in problem.edges:
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
    check()
    if arcs:model.add_circuit(arcs)
    else:model.add_bool_or([])  # Nonempty puzzle tiles require actual gold ports.
    check();stats.update(circuit_nodes=len(port_nodes),circuit_arcs=len(arcs),
                         circuit_variables=len(model.proto.variables),circuit_constraints=len(model.proto.constraints))
    return CircuitModel(model,codes,placements,stats)
