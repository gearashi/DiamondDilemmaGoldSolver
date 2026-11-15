"""Export a reproducible reference-loss CNF and check signed assignments against raw tile endpoints."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from r35_integer_sat import InventorySAT
from r32_gold_bonds import inventory
from r32_gold_loss import read_json, reference_groups
from r32_gold_syndrome import OutsideSyndromes
from search_problem import SearchProblem


from r35_loss_witness import physical_price, raw_sides, reciprocal


def original(tiles):
    problem = SearchProblem(read_json(tiles))
    root = Path(__file__).resolve().parent
    reference, _, all_groups = reference_groups(problem, read_json(root/'docs/R29_INCUMBENT.json')['codes'])
    groups, required, outside = OutsideSyndromes.layout(reference, all_groups, inventory(problem).required)
    if any(set(group) & set(other) for i, group in enumerate(groups) for other in groups[:i]):
        raise ValueError('The charged reference groups overlap.')
    return problem, reference, groups, required, outside


def export(tiles, cap, destination, *, excess=False):
    p, reference, groups, required, outside = original(tiles)
    cnf = InventorySAT(p, groups, required, outside, cap, native=None, excess=excess)
    clauses = cnf.clauses
    variables = max(abs(v) for c in clauses for v in c)
    with destination.open('w', encoding='ascii', newline='\n') as stream:
        stream.write(f'p cnf {variables} {len(clauses)}\n')
        for row in clauses:
            stream.write(' '.join(map(str, row)) + ' 0\n')
    return dict(problem=p.fingerprint, cap=cap, bonds=len(reference), disjoint_groups=len(groups),
                required=len(required), outside=len(outside), excess=excess, variables=variables, clauses=len(clauses),
                cnf_sha256=hashlib.sha256(destination.read_bytes()).hexdigest())



def closes_proper_loop(tiles, pairs):
    adjacency = {}
    for tile, record in enumerate(tiles):
        for (a, i), (b, j) in record['segments']:
            u, v = (3 * tile + a, i), (3 * tile + b, j)
            adjacency[u], adjacency[v] = [v], [u]
    for a, b in pairs:
        for slot in range(1, 12):
            x, y = (a, slot), (b, 12-slot)
            if x in adjacency:
                if y not in adjacency:
                    raise ValueError('A reference seam has mismatching slots.')
                adjacency[x].append(y)
                adjacency[y].append(x)
    visited = set()
    for u in adjacency:
        if u in visited:
            continue
        todo = [u]
        component = set()
        while todo:
            x = todo.pop()
            if x in component:
                continue
            component.add(x)
            todo.extend(adjacency[x])
        visited.update(component)
        if len(component) < len(adjacency) and all(len(adjacency[x]) == 2 for x in component):
            return True
    return False


def parity_conflict(sides, pairs):
    # Count a reciprocal pair of signed classes jointly modulo two. Self-
    # reciprocal classes must also contain an even number of physical sides.
    counts = {}
    for side, (slots, color) in enumerate(sides):
        for phase in (0, 1):
            label = slots, color ^ phase
            key = min(label, reciprocal(label))
            row, const = counts.get(key, (0, 0))
            counts[key] = row ^ (1 << (side // 3)), const ^ (phase == 0)
    equations = list(counts.values())
    for a, b in pairs:
        sa, ca = sides[a]
        sb, cb = sides[b]
        if reciprocal((sa, ca))[0] != sb:
            return True
        equations.append(((1 << (a//3)) ^ (1 << (b//3)), ca ^ cb ^ (len(sa) % 2)))
    basis = {}
    for row, const in equations:
        while row:
            pivot = row & -row
            if pivot not in basis:
                basis[pivot] = row, const
                break
            other, rhs = basis[pivot]
            row, const = row ^ other, const ^ rhs
        else:
            if const:
                return True
    return False


def independent_obligations(p, reference, groups, required, codes):
    sides = raw_sides(p.tiles)
    computed = set()
    for cell, side, adjacent, other in p.edges:
        a = 3*(codes[cell]//3) + (side-codes[cell]%3)%3
        b = 3*(codes[adjacent]//3) + (other-codes[adjacent]%3)%3
        if sides[b][0] == reciprocal(sides[a])[0]:
            computed.add(tuple(sorted((a, b))))
    if computed != reference or len(set(c//3 for c in codes)) != p.n:
        raise AssertionError('The oriented reference seams do not reconstruct.')
    independent = []
    for group in groups:
        cut = closes_proper_loop(p.tiles, group)
        parity = parity_conflict(sides, group)
        if not (cut or parity):
            raise AssertionError(('Uncertified conflict group', group))
        independent.append(dict(size=len(group), proper_loop=cut, parity_conflict=parity))
    forced = []
    for a, b in sorted(required):
        key = {sides[a][0], sides[b][0]}
        members = frozenset(i for i, (slots, _) in enumerate(sides) if slots in key)
        # Limit this proof helper to the small reciprocal classes claimed forced.
        if len(members) > 16:
            raise ValueError('Large reciprocal class requires another independent proof.')
        allowed = {(u, v) for u in members for v in members if u < v and u//3 != v//3
                   and sides[v][0] == reciprocal(sides[u])[0]
                   and not closes_proper_loop(p.tiles, [(u, v)])}

        def count_matchings(rest, forbidden):
            if not rest:
                return 1
            u = min(rest)
            return sum(count_matchings(rest-{u,v}, forbidden) for v in rest-{u}
                       if (u,v) != forbidden and (u,v) in allowed)

        n = count_matchings(members, None)
        without = count_matchings(members, (a,b))
        if not n or without:
            raise AssertionError(('Uncertified required bond', (a,b), n, without))
        forced.append(dict(bond=[a,b], class_size=len(members), matchings=n, without=without))
    return dict(tiles=p.n, sides=len(sides), reference=len(computed), groups=independent, forced=forced)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('action', choices=['export', 'price'])
    cli.add_argument('--tiles', type=Path, required=True)
    cli.add_argument('--cap', type=int, required=True)
    cli.add_argument('--excess', action='store_true')
    cli.add_argument('--output', type=Path)
    cli.add_argument('--phase', type=lambda x: int(x, 0))
    args = cli.parse_args()
    if args.action == 'export':
        print(json.dumps(export(args.tiles, args.cap, args.output, excess=args.excess)))
    else:
        p, ref, groups, req, out = original(args.tiles)
        price = physical_price(p.tiles, ref, groups, req, args.phase)
        print(json.dumps(dict(cost=None if price is None else price[0], phase=hex(args.phase))))
