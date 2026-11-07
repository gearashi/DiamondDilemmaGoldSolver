"""Exact bounded compatibility of completed regional summaries.

All relations are scoped to the complete current domain vector.  A stage only
publishes its supported rows after every combination in that stage has been
examined.  Local tables remain immutable.  Resource exhaustion therefore loses
strength, never a solution.  A global combination checks inventory AND strand
interfaces; disconnected regions are never assumed independent beforehand.
"""
from __future__ import annotations
from collections import OrderedDict
from itertools import combinations
from constraint_pruning import values
from constraint_separator import _Budget
from validator import reverse_mask

STAT_KEYS = ('region_network_checks', 'region_network_cache_hits', 'region_network_live_hits',
    'region_network_comparisons', 'region_network_rounds',
    'region_network_summaries_removed', 'region_network_prunes',
    'region_network_budget_skips', 'region_triple_checks',
    'region_inventory_checks', 'region_outside_path_rejections',
    'region_overlap_checks', 'region_overlap_prunes',
    'region_diagram_nodes', 'region_diagram_rows', 'region_component_joins',
    'region_symbolic_states', 'region_symbolic_hits', 'region_inventory_zdd_nodes',
    'region_inventory_zdd_operations', 'region_inventory_subset_operations')


def joined_signature(owner, left, ls, right, rs, domains, check):
    """Exact disjoint union; return None only for a proved incompatibility."""
    if set(left) & set(right):
        raise ValueError('Joining summaries requires disjoint cells.')
    used_l, masks_l, pairs_l, cycles_l = ls
    used_r, masks_r, pairs_r, cycles_r = rs
    if used_l & used_r:
        return None
    le, re = owner._frontier(set(left)), owner._frontier(set(right))
    lm, rm = dict(zip(le, masks_l)), dict(zip(re, masks_r))
    shared = set(lm) & set(rm)
    if any(lm[e] != rm[e] for e in shared):
        return None
    members = set(left) | set(right)
    frontier = {**lm, **rm}
    for edge in shared:
        del frontier[edge]
    boundary = tuple((c, domains[c]) for c in sorted(
        set().union(*(owner.neighbors[c] for c in members)) - members))
    if not owner._outside_possible(used_l | used_r, frontier, boundary, members):
        return None
    mates = {p:q for pair in (*pairs_l, *pairs_r) for p,q in (pair, pair[::-1])}
    cycles = cycles_l + cycles_r
    for eid in sorted(shared):
        check()
        a,sa,b,sb = owner.problem.edges[eid]
        for position in values(lm[eid]):
            cycles += owner._join(mates, 33*a+11*sa+position, 33*b+11*sb+10-position)
    if owner.target == 'gold' and cycles and not (
            len(members) == owner.problem.n and cycles == 1 and not mates):
        return None
    edges = owner._frontier(members)
    return (used_l | used_r, tuple(frontier[e] for e in edges),
            tuple(sorted((a,b) for a,b in mates.items() if a < b)), cycles)


def assignment_signature(owner, cells, codes, domains, check):
    """Evaluate one concrete regional history, preserving every strand end."""
    members = set(cells)
    chosen = dict(zip(cells, codes))
    if len({c//3 for c in codes}) != len(codes):
        return None
    used = sum(1 << (code//3) for code in codes)
    frontier, mates, cycles = {}, {}, 0
    for cell,code in chosen.items():
        check()
        row = owner.masks[code]
        if any(owner.problem.neighbors[cell,s] < 0 and row[s] for s in range(3)):
            return None
        for p,q in owner.pairs[code]:
            p += 33*cell; q += 33*cell
            mates[p] = q; mates[q] = p
    for eid,(a,sa,b,sb) in enumerate(owner.problem.edges):
        if a not in members and b not in members:
            continue
        if a in members and b in members:
            mask = owner.masks[chosen[a]][sa]
            if mask != reverse_mask(owner.masks[chosen[b]][sb]):
                return None
            for position in values(mask):
                cycles += owner._join(mates, 33*a+11*sa+position, 33*b+11*sb+10-position)
        else:
            cell,side = (a,sa) if a in members else (b,sb)
            mask = owner.masks[chosen[cell]][side]
            frontier[eid] = mask if cell == a else reverse_mask(mask)
    if owner.target == 'gold' and cycles and not (
            len(cells) == owner.problem.n and cycles == 1 and not mates):
        return None
    boundary = tuple((c, domains[c]) for c in sorted(
        set().union(*(owner.neighbors[c] for c in cells)) - members))
    if not owner._outside_possible(used, frontier, boundary, members):
        return None
    return (used, tuple(frontier[e] for e in owner._frontier(members)),
            tuple(sorted((a,b) for a,b in mates.items() if a < b)), cycles)


class ExactRegionDiagram:
    """Reduced ordered multi-valued diagram retaining exact cell assignments.

    Equal suffix relations share one node.  Terminal nodes retain complete
    boundary signatures.  This is deliberately distinct from support-union
    summaries: exact shared-cell rows are required for overlapping regions.
    """
    def __init__(self, owner, cells, domains, check, max_rows=4096, max_steps=200000):
        self.cells = tuple(sorted(cells))
        self.nodes, self.terminals, self.rows, self.steps = {}, {}, 0, 0
        intern, rows, assignment = {}, [], {}
        order = sorted(self.cells, key=lambda c:(domains[c].bit_count(), c))
        def walk(depth, used):
            check()
            if depth == len(order):
                codes = tuple(assignment[c] for c in self.cells)
                signature = assignment_signature(owner, self.cells, codes, domains, check)
                if signature is not None:
                    if len(rows) >= max_rows:
                        raise _Budget
                    rows.append((codes, signature))
                return
            cell = order[depth]
            for code in values(domains[cell]):
                self.steps += 1
                if self.steps > max_steps:
                    raise _Budget
                if used >> (code//3) & 1:
                    continue
                if any(other in assignment and owner.masks[code][side] !=
                       reverse_mask(owner.masks[assignment[other]][other_side])
                       for other,side,other_side,_ in owner.adj[cell]):
                    continue
                assignment[cell] = code
                walk(depth+1, used | (1 << (code//3)))
                del assignment[cell]
        walk(0, 0)
        self.rows = len(rows)
        def build(level, selected):
            check()
            if level == len(self.cells):
                signature = selected[0][1]
                key = ('terminal', signature)
                if key not in intern:
                    ident = len(intern); intern[key] = ident
                    self.terminals[ident] = signature
                return intern[key]
            groups = {}
            for codes,signature in selected:
                groups.setdefault(codes[level], []).append((codes,signature))
            edges = tuple((code,build(level+1,group)) for code,group in sorted(groups.items()))
            key = (level,edges)
            if key not in intern:
                ident = len(intern); intern[key] = ident; self.nodes[ident] = (level,edges)
            return intern[key]
        self.root = build(0,rows) if rows else None

    def histories(self, check=lambda:None):
        def walk(node, codes):
            check()
            if node in self.terminals:
                yield tuple(codes), self.terminals[node]
                return
            _,edges = self.nodes[node]
            for code,child in edges:
                yield from walk(child, (*codes,code))
        if self.root is not None:
            yield from walk(self.root, ())


class RegionNetwork:
    def __init__(self, owner):
        self.owner = owner
        self.max_steps = 50000
        self.max_tables = 4
        self.max_rows = 4096
        self.cache_limit = 16
        self.cache = OrderedDict()
        self.live_cache = OrderedDict()
        self.max_live_rows = 8192
        self.triples = True
        self.inventories = True
        self.outside_paths = True
        self.last = None
        self.direct_diagrams = False
        self.symbolic_inventory = False
        self.subset_inventory = False
        self._diagram_sessions = OrderedDict()
        self.stats = dict.fromkeys(STAT_KEYS, 0)

    def _publish(self, domains, supports, check, overlap=False):
        changed = False
        dependencies = tuple(range(self.owner.problem.n))
        for cell,support in supports.items():
            check()
            old = domains[cell]; new = old & support
            if old != new:
                domains[cell] = new; changed = True
                self.stats['region_overlap_prunes' if overlap else 'region_network_prunes'] += (old ^ new).bit_count()
                if self.owner.on_narrow is not None:
                    self.owner.on_narrow(cell,new,dependencies)
                if not new:
                    self.owner.note_conflict(tuple(supports))
                    break
        return changed

    def _tick(self, check):
        self._steps += 1
        self.stats['region_network_comparisons'] += 1
        if self._steps > self.max_steps:
            raise _Budget
        if self._steps % 16 == 0:
            check()

    def _fixed_compatible(self, cells, signature, domains, check):
        if not self.outside_paths:
            return True
        outside = tuple(c for c,d in enumerate(domains) if c not in cells and d.bit_count() == 1)
        if not outside:
            return True
        # Scope is the full domain vector, including remote fixed chains.
        codes = tuple(domains[c].bit_length()-1 for c in outside)
        fixed = assignment_signature(self.owner,outside,codes,domains,check)
        valid = fixed is not None and joined_signature(self.owner,cells,signature,outside,fixed,domains,check) is not None
        if not valid:
            self.stats['region_outside_path_rejections'] += 1
        return valid

    def prune(self, domains, regions, check=lambda:None):
        self.last = None
        if not all(domains):
            return False
        regions = tuple(dict.fromkeys(tuple(sorted(set(r))) for r in regions))[:self.max_tables]
        if not regions or any(not r for r in regions) or any(set(a)&set(b) for a,b in combinations(regions,2)):
            return False
        check(); self.stats['region_network_checks'] += 1
        key = (tuple(domains),regions,self.triples,self.inventories,self.outside_paths)
        cached = self.cache.get(key)
        if cached is not None:
            self.stats['region_network_cache_hits'] += 1
            self.cache.move_to_end(key)
            self.last = cached[1]
            return self._publish(domains,cached[0],check)
        tables = []
        for cells in regions:
            _,result = self.owner._completed_table(domains,cells,check)
            if result is None:
                return False
            tables.append(result['table'])
        active = [set(t) for t in tables]
        for index,cells in enumerate(regions):
            live_key = tuple(domains),cells
            if live_key in self.live_cache:
                active[index].intersection_update(self.live_cache[live_key])
                self.stats['region_network_live_hits'] += 1
                self.live_cache.move_to_end(live_key)
        original_count = sum(map(len,active))
        self._steps = 0
        complete = True
        try:
            # Fully check each unary stage before replacing its active rows.
            for index,cells in enumerate(regions):
                allowed = set()
                for signature in sorted(active[index]):
                    self._tick(check)
                    if self._fixed_compatible(cells,signature,domains,check):
                        allowed.add(signature)
                active[index] = allowed
            changed = True
            while changed:
                changed = False
                self.stats['region_network_rounds'] += 1
                for i,j in combinations(range(len(regions)),2):
                    ai,aj = set(),set()
                    for left in sorted(active[i]):
                        for right in sorted(active[j]):
                            self._tick(check)
                            if joined_signature(self.owner,regions[i],left,regions[j],right,domains,check) is not None:
                                ai.add(left); aj.add(right)
                    if ai != active[i] or aj != active[j]:
                        active[i],active[j] = ai,aj; changed = True
                if self.triples:
                    for i,j,k in combinations(range(len(regions)),3):
                        self.stats['region_triple_checks'] += 1
                        allowed = [set(),set(),set()]
                        for si in sorted(active[i]):
                            for sj in sorted(active[j]):
                                self._tick(check)
                                combined = joined_signature(self.owner,regions[i],si,regions[j],sj,domains,check)
                                if combined is None:
                                    continue
                                cells = tuple(sorted((*regions[i],*regions[j])))
                                for sk in sorted(active[k]):
                                    self._tick(check)
                                    if joined_signature(self.owner,cells,combined,regions[k],sk,domains,check) is not None:
                                        allowed[0].add(si); allowed[1].add(sj); allowed[2].add(sk)
                        for index,supported in zip((i,j,k),allowed):
                            if supported != active[index]:
                                active[index] = supported; changed = True
                if self.inventories and len(regions) >= 3:
                    # Exact allocation of WHOLE inventories.  Completed failed
                    # suffixes are memoized; masks are not split into tile unions.
                    self.stats['region_inventory_checks'] += 1
                    groups = [set(s[0] for s in row) for row in active]
                    if self.subset_inventory:
                        from constraint_region_symbolic import subset_inventory_supports
                        supported,work=subset_inventory_supports(groups,check)
                        self.stats['region_inventory_subset_operations']+=work['operations']
                    elif self.symbolic_inventory:
                        from constraint_region_symbolic import inventory_supports
                        supported,work=inventory_supports(groups,check,max_nodes=self.max_steps)
                        self.stats['region_inventory_zdd_nodes']+=work['nodes']
                        self.stats['region_inventory_zdd_operations']+=work['operations']
                    else:
                        supported = [set() for _ in groups]
                        memo = {}
                        def allocate(index,used):
                            self._tick(check)
                            cache_key = index,used
                            if cache_key in memo:
                                return memo[cache_key]
                            if index == len(groups):
                                return True
                            feasible = False
                            for inventory in sorted(groups[index]):
                                if inventory & used:
                                    continue
                                if allocate(index+1,used|inventory):
                                    supported[index].add(inventory); feasible = True
                            memo[cache_key] = feasible
                            return feasible
                        allocate(0,0)
                    for index,allowed in enumerate(supported):
                        selected = {s for s in active[index] if s[0] in allowed}
                        if selected != active[index]:
                            active[index] = selected; changed = True
        except _Budget:
            complete = False
            self.stats['region_network_budget_skips'] += 1
        # Every earlier completed stage remains a valid deduction after a cap.
        supports = {}
        for cells,table,selected in zip(regions,tables,active):
            for offset,cell in enumerate(cells):
                supports[cell] = 0
                for signature in selected:
                    supports[cell] |= table[signature][0][offset]
        check()
        for cells,selected in zip(regions,active):
            if len(selected)>self.max_live_rows:continue
            live_key = tuple(domains),cells
            self.live_cache[live_key] = frozenset(selected)
            while len(self.live_cache)>16 or sum(map(len,self.live_cache.values()))>self.max_live_rows:
                self.live_cache.popitem(last=False)
        removed = original_count - sum(map(len,active))
        self.stats['region_network_summaries_removed'] += removed
        self.last = {'complete':complete,'regions':regions,'steps':self._steps,
                     'summaries_removed':removed,'live_rows':tuple(map(len,active))}
        if complete and self.cache_limit > 0:
            self.cache[key] = supports,self.last
            while len(self.cache) > self.cache_limit:
                self.cache.popitem(last=False)
        return self._publish(domains,supports,check)

    def _direct_diagram(self,cells,domains,check):
        from constraint_region_symbolic import DirectRegionDiagram,DirectRegionSession
        key=(tuple(cells),tuple(domains))
        session=self._diagram_sessions.get(key)
        if session is None:
            session=DirectRegionSession(self.owner,cells,domains,self.max_rows,self.max_steps)
            self._diagram_sessions[key]=session
            while len(self._diagram_sessions)>4:self._diagram_sessions.popitem(last=False)
        self._diagram_sessions.move_to_end(key)
        try:
            while not session.advance(256,check):pass
        except _Budget:
            self._diagram_sessions.pop(key,None)
            raise
        # Cancellation leaves the suspended session available for the identical
        # immutable query on the next work slice. Only complete DAGs escape.
        diagram=object.__new__(DirectRegionDiagram);diagram.__dict__.update(session.result())
        return diagram

    def prune_overlap(self, domains, left, right, check=lambda:None):
        if not all(domains):
            return False
        left,right = tuple(sorted(set(left))),tuple(sorted(set(right)))
        overlap = set(left)&set(right)
        if not overlap or not set(left)-overlap or not set(right)-overlap:
            return False
        self.stats['region_overlap_checks'] += 1; self._steps = 0
        try:
            if self.direct_diagrams:
                from constraint_region_symbolic import DirectRegionDiagram,join_diagrams
                ld=self._direct_diagram(left,domains,check)
                rd=self._direct_diagram(right,domains,check)
                supports,work=join_diagrams(self.owner,ld,rd,domains,check,self.max_steps)
                self.stats['region_symbolic_states']+=ld.states+rd.states+work['states']
                self.stats['region_symbolic_hits']+=ld.memo_hits+rd.memo_hits+work['memo_hits']
                self.stats['region_diagram_nodes']+=len(ld.nodes)+len(rd.nodes)
                self.stats['region_diagram_rows']+=ld.rows+rd.rows
            else:
                ld = ExactRegionDiagram(self.owner,left,domains,check,self.max_rows,self.max_steps)
                rd = ExactRegionDiagram(self.owner,right,domains,check,self.max_rows,self.max_steps)
                self.stats['region_diagram_nodes'] += len(ld.nodes)+len(rd.nodes)
                self.stats['region_diagram_rows'] += ld.rows+rd.rows
                index = {}
                shared = tuple(sorted(overlap))
                lo = tuple(left.index(c) for c in shared); ro = tuple(right.index(c) for c in shared)
                for codes,signature in rd.histories(check):
                    index.setdefault(tuple(codes[i] for i in ro), []).append(codes)
                supports = {c:0 for c in set(left)|set(right)}
                members = tuple(sorted(supports))
                for lc,_ in ld.histories(check):
                    for rc in index.get(tuple(lc[i] for i in lo), ()):
                        self._tick(check)
                        chosen = dict(zip(left,lc)); chosen.update(zip(right,rc))
                        codes = tuple(chosen[c] for c in members)
                        if assignment_signature(self.owner,members,codes,domains,check) is not None:
                            for cell,code in chosen.items():
                                supports[cell] |= 1 << code
        except _Budget:
            self.stats['region_network_budget_skips'] += 1
            return False
        return self._publish(domains,supports,check,overlap=True)

    def prune_components(self, domains, components, check=lambda:None):
        """Exact AND/OR interface composition, including physical inventories.

        Each component table represents an AND subproblem only conditional on
        its inventory, boundary masks and endpoint pairing.  OR alternatives
        are joined before supports can be published.  Failed suffix joins are
        memoized by the complete already-composed signature and component index.
        """
        components = tuple(tuple(sorted(set(c))) for c in components)
        if not all(domains) or not components or any(set(a)&set(b) for a,b in combinations(components,2)):
            return False
        tables = []
        for cells in components:
            _,result = self.owner._completed_table(domains,cells,check)
            if result is None:
                return False
            tables.append(result['table'])
        supports = {c:0 for cells in components for c in cells}
        self._steps = 0; failed = set()
        def walk(index,cells,signature,history):
            self._tick(check)
            if index == len(components):
                if not self._fixed_compatible(cells,signature,domains,check):return False
                for part,selected,table in zip(components,history,tables):
                    for cell,mask in zip(part,table[selected][0]):
                        supports[cell] |= mask
                return True
            key = index,cells,signature
            if key in failed:
                return False
            any_support = False
            for selected in tables[index]:
                self._tick(check)
                combined = selected if index == 0 else joined_signature(
                    self.owner,cells,signature,components[index],selected,domains,check)
                if combined is None:
                    continue
                self.stats['region_component_joins'] += 1
                any_support |= walk(index+1,tuple(sorted((*cells,*components[index]))),combined,(*history,selected))
            if not any_support:
                failed.add(key)
            return any_support
        try:
            walk(0,(),None,())
        except _Budget:
            self.stats['region_network_budget_skips'] += 1
            return False
        return self._publish(domains,supports,check)
