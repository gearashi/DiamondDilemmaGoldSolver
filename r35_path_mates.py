"""Infer extra physical seams shared by every edge-compatible embedding of a rigid component."""
from collections import OrderedDict, defaultdict
from time import monotonic

from r27_structure_filters import Union
from r35_geographic_packing_check import component_poses
from validator import reverse_mask


class PathMates:
    def __init__(self, problem):
        self.n = problem.n
        self.incidence = {}
        for a,s,b,t in problem.edges:
            self.incidence[a,s] = b,t
            self.incidence[b,t] = a,s
        self.mask = tuple(int(x) for row in problem.masks[::3] for x in row)
        self.reverse = tuple(reverse_mask(x) for x in self.mask)

    def patterns(self, bonds, domains=None, deadline=float("inf")):
        """Return all distinct sets of extra compatible internal joins and count labelled roots."""
        bonds = frozenset(bonds)
        geometric = matched = 0
        patterns = set()
        for _, poses in component_poses(self.n,self.incidence,bonds):
            if monotonic() >= deadline:
                raise TimeoutError
            if domains is not None and any(not domains[cell] & (1<<(3*tile+turn))
                                           for tile,(cell,turn) in poses.items()):
                continue
            geometric += 1
            occupied = {cell:(tile,turn) for tile,(cell,turn) in poses.items()}
            extras = set()
            valid = True
            for tile,(cell,turn) in poses.items():
                for side in range(3):
                    adjacent = self.incidence.get((cell,(side+turn)%3))
                    if adjacent is None or adjacent[0] not in occupied:
                        continue
                    neighbor,rotation = occupied[adjacent[0]]
                    if neighbor <= tile:
                        continue
                    a = 3*tile+side
                    b = 3*neighbor+(adjacent[1]-rotation)%3
                    if self.mask[a] != self.reverse[b]:
                        valid = False
                        break
                    if (a,b) not in bonds:
                        extras.add((a,b))
                if not valid:
                    break
            if valid:
                matched += 1
                patterns.add(frozenset(extras))
        if monotonic() >= deadline:
            raise TimeoutError
        return patterns, geometric, matched


def alternatives(patterns, width=3):
    """Return inclusion-minimal seam disjunctions up to width that hit every pattern."""
    if not patterns or any(not row for row in patterns):
        return ()
    rows = sorted(patterns,key=lambda q:(len(q),sorted(q)))
    covers = set()
    def extend(chosen):
        if any(cover<=chosen for cover in covers):
            return
        missing = next((row for row in rows if not row&chosen),None)
        if missing is None:
            covers.difference_update([old for old in covers if chosen<old])
            covers.add(chosen)
        elif len(chosen)<width:
            for bond in sorted(missing):
                extend(chosen|{bond})
    extend(frozenset())
    return tuple(sorted(covers,key=lambda c:(len(c),sorted(c))))


def propagate(problem, domains, required, forbidden, deadline=float("inf")):
    wanted = frozenset(tuple(sorted(e)) for e in required)
    blocked = frozenset(tuple(sorted(e)) for e in forbidden)
    checker = PathMates(problem)
    groups = defaultdict(set)
    union = Union(problem.n)
    more = set()
    absent = set()
    tables = []
    stats = dict(components=0, geometric_roots=0, compatible_roots=0, distinct_patterns=0)
    def finish(status):
        stats.update(added_required=len(more),added_forbidden=len(absent))
        return status,more,absent,tables,stats
    if len({d for e in wanted for d in e}) != 2*len(wanted):
        return finish("SKIPPED")
    for a,b in wanted:
        if a//3==b//3 or checker.mask[a]!=checker.reverse[b]:
            return finish("SKIPPED")
        union.join(a//3,b//3)
    for a,b in wanted:
        groups[union.find(a//3)].add((a,b))
    for group in sorted(groups.values(),key=lambda g:(-len(g),sorted(g))):
        try:
            patterns,geo,matched = checker.patterns(group,domains,deadline)
        except TimeoutError:
            return finish("LIMIT")
        stats["components"] += 1
        stats["geometric_roots"] += geo
        stats["compatible_roots"] += matched
        survivors = [r for r in patterns if not r&blocked]
        stats["distinct_patterns"] += len(survivors)
        if not survivors or group&blocked:
            return finish("INFEASIBLE")
        common = frozenset.intersection(*survivors)
        possible = frozenset.union(*survivors)
        more.update(common)
        taken = {d for pair in group for d in pair}
        free = sorted({3*(d//3)+s for d in taken for s in range(3)}-taken)
        for i,a in enumerate(free):
            for b in free[i+1:]:
                if (a//3!=b//3 and checker.mask[a]==checker.reverse[b]
                        and (a,b) not in possible and (a,b) not in blocked):
                    absent.add((a,b))
        variables = tuple(sorted(possible-common))
        if variables:
            tables.append((variables,survivors))
    return finish("SURVIVING")


def post_tables(check, tables):
    posted = rows = unsupported = 0
    projected = check.blank_projection.allowed if check.blank_projection is not None else ()
    for variables, patterns in tables:
        columns = [bond for bond in variables if bond in check.mate]
        unavailable = set(variables).difference(columns, projected)
        surviving = [row for row in patterns if not unavailable.intersection(row)]
        unsupported += len(patterns) - len(surviving)
        if not surviving:
            check.m.add_bool_or([])
            posted += 1
            break
        if not columns:
            continue
        values = {tuple(int(bond in row) for bond in columns) for row in surviving}
        if len(values) == 1 << len(columns):
            continue
        check.m.add_allowed_assignments([check.mate[bond] for bond in columns], sorted(values))
        posted += 1
        rows += len(values)
    return dict(tables=posted, rows=rows, unsupported_patterns=unsupported)


def _circuit_replay(problem):
    # Match the exact inputs used by placement propagation and by physical loop tracing.
    artwork = tuple(tuple(tuple((s,k) for s,k in segment) for segment in tile["segments"])
                    for tile in problem.tiles)
    key = (problem.n, problem.total_segments, tuple(map(tuple,problem.edges)),
           problem.masks[::3].tobytes(), artwork)
    old = getattr(problem, "_path_circuit_replay", None)
    if old is None or old[0] != key:
        from r32_gold_bonds import endpoints
        from r35_loss_witness import raw_sides
        old = (key, PathMates(problem), endpoints(problem),
               [slots for slots,_ in raw_sides(problem.tiles)], OrderedDict())
        problem._path_circuit_replay = old
    return old[1:]


def verified_circuits(problem, hints, allowed=None):
    """Substitute a geographically forced extra mate into a strict physical loop."""
    from r32_gold_bonds import loop_groups
    from r35_geographic_replay import replay_for

    if not hints:
        return ()
    normalizer = replay_for(problem)
    checker, arcs, sides, cached = _circuit_replay(problem)
    common_by_path = {}
    proved_loops = {}
    groups = set()
    for hint in hints:
        if not isinstance(hint, dict):
            continue
        path = normalizer.normalize(hint.get("path", ()))
        circuit = normalizer.normalize(hint.get("loop", ()))
        single = normalizer.normalize([hint.get("mate")])
        if not path or not circuit or not single or not single <= circuit or path & single:
            continue
        group = path | (circuit - single)
        if allowed is not None and not group <= allowed:
            continue
        record = path, circuit, single
        if record in cached:
            cached.move_to_end(record)
            groups.add(group)
            continue
        if circuit not in proved_loops:
            proved_loops[circuit] = circuit in loop_groups(arcs, sides, circuit, problem.total_segments)
        if not proved_loops[circuit]:
            continue
        if path not in common_by_path:
            try:
                patterns, _, _ = checker.patterns(path)
            except ValueError:
                patterns = set()
            common_by_path[path] = frozenset.intersection(*patterns) if patterns else frozenset()
        if single <= common_by_path[path]:
            if len(cached) >= 128:
                cached.popitem(last=False)
            cached[record] = True
            groups.add(group)
    return tuple(sorted(groups, key=lambda g: (len(g), sorted(g))))


def circuit_departures(problem, reference, required, hints):
    required = frozenset(required)
    rows = verified_circuits(problem, hints, frozenset(reference) | required)
    return tuple(sorted({row-required for row in rows}, key=lambda g: (len(g),sorted(g))))
