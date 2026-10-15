"""Process-local finite-domain lazy clause generation with first-UIP analysis.

A literal ``(cell, mask)`` means the assigned orientation belongs to ``mask``.
A learned nogood forbids all its literals being true together. Propagation
removes the last unknown mask; it does not require singleton tile placements.
Facts are implications from earlier domain facts, with explicit decision levels.
Resolving the latest current-level implication until one remains retains an
intermediate UIP fact instead of flattening every reason into decisions.

Callers must supply SOUND rule antecedents and conflicts. Global rules may use
all pre-call domain facts. Reconstructing a checkpoint is conservative only when
its restrictions are already known sound under the supplied active path (the
ordinary solver checkpoint trust contract); this module never deserializes proof
nodes or accepts serialized learned clauses. Stores bind exact effective roots,
so worker-restricted proofs cannot escape to sibling jobs or a wider puzzle.
"""
from __future__ import annotations
from dataclasses import dataclass
import heapq

STAT_KEYS = ('lcg_nodes', 'lcg_rebases', 'lcg_conflicts', 'lcg_resolutions',
             'lcg_learned', 'lcg_subsumed', 'lcg_evictions', 'lcg_checks',
             'lcg_prunes', 'lcg_hits', 'lcg_budget_skips',
             'lcg_reason_expansions', 'lcg_reason_parent_visits',
             'lcg_frontier_dominated', 'lcg_clause_skips',
             'lcg_assertions', 'lcg_asserted_values', 'lcg_asserting_jumps',
             'lcg_asserting_levels', 'lcg_assertion_fallbacks')
MAX_FACTS = 65536
MAX_CLAUSES = 1024
MAX_LITERALS = 4096


def _masks(domains, initial=None):
    values = tuple(domains)
    if not values or any(type(v) is not int or v < 0 for v in values):
        raise ValueError('LCG domains must be nonnegative integer masks.')
    if initial is not None and (len(values) != len(initial) or
                               any(v & ~old for v, old in zip(values, initial))):
        raise ValueError('LCG domains escaped their effective root.')
    return values


def _binding(fingerprint, target, initial):
    if not isinstance(fingerprint, str) or not fingerprint or target not in ('gold', 'edges'):
        raise ValueError('Invalid LCG puzzle or target binding.')
    return fingerprint, target, _masks(initial)


@dataclass(frozen=True, slots=True)
class _SharedReason:
    """Immutable pre-rule fact snapshot shared by all simultaneous outputs."""
    refs: tuple
    level: int


@dataclass(frozen=True, slots=True, eq=False)
class _LazyParents:
    shared: _SharedReason
    prior: int = -1

    def __iter__(self):
        yield from self.shared.refs
        if self.prior >= 0:
            yield self.prior

    def __eq__(self, other):
        # Keep diagnostic/test access compatible with the old tuple parents.
        return tuple(self) == tuple(other)


@dataclass(frozen=True, slots=True)
class Fact:
    cell: int
    mask: int
    level: int
    parents: tuple | _LazyParents
    rule: str
    decision: bool = False


@dataclass(frozen=True)
class LearnedClause:
    binding: tuple
    literals: tuple
    uip: tuple | None
    level: int
    backjump_level: int
    resolutions: int
    reason_expansions: int = 0
    parent_visits: int = 0
    dominated: int = 0
    # These are process-local proof results, never a checkpoint file format.


class LCGState:
    def __init__(self, initial_domains, fingerprint, target='gold', *, max_facts=8192):
        self.binding = _binding(fingerprint, target, initial_domains)
        self.initial_domains = self.binding[2]
        if type(max_facts) is not int or not len(self.initial_domains) <= max_facts <= MAX_FACTS:
            raise ValueError('Invalid LCG fact budget.')
        self.max_facts = max_facts
        self.domains = list(self.initial_domains)
        self.facts = [Fact(c, v, 0, (), 'root') for c, v in enumerate(self.domains)]
        self.current = list(range(len(self.domains)))
        self.level = 0
        self.enabled = True
        self.failure = None
        self.budget_skips = 0

    def _cell(self, cell):
        if type(cell) is not int or not 0 <= cell < len(self.domains):
            raise ValueError('Invalid LCG cell.')
        return cell

    def _parents(self, dependencies):
        return tuple(dict.fromkeys(self.current[self._cell(c)] for c in dependencies))

    def _append(self, cell, value, parents, rule, *, decision=False, level=None):
        if not self.enabled:
            return 0
        if len(self.facts) >= self.max_facts:
            # Losing explanation capacity never turns an interrupted or unknown
            # computation into an infeasibility proof. Search continues normally.
            self.enabled = False
            self.budget_skips += 1
            return 0
        if level is None:
            level = max((self.facts[p].level for p in parents), default=0)
        self.facts.append(Fact(cell, value, level, parents, str(rule), decision))
        self.current[cell] = len(self.facts)-1
        return 1

    def fork(self, cell, mask):
        self._cell(cell)
        if type(mask) is not int or mask <= 0 or not (value := self.domains[cell] & mask):
            raise ValueError('LCG decision does not intersect its domain.')
        child = object.__new__(LCGState)
        child.__dict__ = self.__dict__.copy()
        child.domains = self.domains.copy()
        child.current = self.current.copy()
        child.failure = None
        child.level = self.level + 1
        child.budget_skips = 0
        # As in DFS, the preceding descendant must be discarded before a new
        # sibling is forked. Root/ancestor references point only into this prefix.
        del self.facts[max(self.current, default=-1)+1:]
        child._append(cell, value, (child.current[cell],), 'decision',
                      decision=True, level=child.level)
        child.domains[cell] = value
        return child

    def record_cell(self, cell, value, dependencies, rule):
        self._cell(cell)
        if type(value) is not int or value < 0 or value & ~self.domains[cell]:
            raise ValueError('LCG facts cannot widen domains.')
        if value == self.domains[cell]:
            return 0
        parents = tuple(dict.fromkeys((*self._parents(dependencies), self.current[cell])))
        count = self._append(cell, value, parents, rule)
        self.domains[cell] = value
        return count

    def record(self, domains, dependencies, rule):
        values = _masks(domains, self.domains)
        outputs = [(c, v) for c, (old, v) in enumerate(zip(self.domains, values)) if v != old]
        if not outputs:
            return 0  # Most bulk wrappers follow precise, already-recorded events.
        if not self.enabled:
            self.domains = list(values)
            return 0
        parents = self._parents(dependencies)  # Capture before any output exists.
        shared = _SharedReason(parents, max((self.facts[p].level for p in parents), default=0))
        captured = set(parents)
        changed = 0
        for cell, value in outputs:
            prior = self.current[cell]
            refs = _LazyParents(shared, -1 if prior in captured else prior)
            level = max(shared.level, self.facts[prior].level)
            changed += self._append(cell, value, refs, rule, level=level)
        self.domains = list(values)
        return changed

    @classmethod
    def reconstruct(cls, initial_domains, domains, decisions=(), *, fingerprint,
                    target='gold', max_facts=8192):
        """Rebase completed SOUND restrictions using all actual path decisions.

        Unlike a serialized explanation graph, only domains and choices enter.
        Every rebased deduction retains all active choices; no restriction is
        silently assigned to an earlier decision level. Interrupted rules must
        have committed only sound narrowings, as required by ordinary DFS.
        """
        state = cls(initial_domains, fingerprint, target, max_facts=max_facts)
        values = _masks(domains, state.initial_domains)
        refs = []
        for pair in decisions:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError('Invalid LCG path decision.')
            cell, mask = pair
            state = state.fork(cell, mask)
            refs.append(state.current[cell])
        _masks(values, state.domains)
        if state.enabled:
            shared = _SharedReason(tuple(refs), max((state.facts[p].level for p in refs), default=0))
            captured = set(refs)
            for cell, value in enumerate(values):
                if value != state.domains[cell]:
                    prior = state.current[cell]
                    parents = _LazyParents(shared, -1 if prior in captured else prior)
                    state._append(cell, value, parents, 'rebased completed restriction',
                                  level=max(shared.level, state.facts[prior].level))
        state.domains = list(values)
        return state

    def first_uip(self, cells=None):
        """Resolve a SOUND conflict and return its first-UIP mask nogood.

        ``cells`` names the current facts which together contradict a puzzle
        rule (e.g. the placements of a proper closed Gold loop). If omitted,
        empty domains provide the conflict. None means unavailable, not UNSAT.
        """
        if not self.enabled:
            return None
        if cells is None:
            cells = self.failure
        if cells is None:
            cells = [c for c, mask in enumerate(self.domains) if not mask]
            if not cells:
                return None
        # Every fact on this active path is nested with older facts of its
        # cell. Keep only the strongest one in the frontier: A AND B == A when
        # A's mask is a subset of B's. Root facts are unconditional axioms and
        # can be discharged on insertion without following their ancestry.
        pending, counts, order, empty = {}, {}, [], []
        resolutions = expansions = parent_visits = dominated = 0

        def add(index):
            nonlocal dominated
            fact = self.facts[index]
            if fact.level == 0:
                return
            previous = pending.get(fact.cell)
            if previous is not None:
                other = self.facts[previous]
                if not other.mask & ~fact.mask:
                    dominated += 1
                    return
                if fact.mask & ~other.mask:
                    raise ValueError('LCG active-path facts are not nested.')
                counts[other.level] -= 1
                dominated += 1
            pending[fact.cell] = index
            counts[fact.level] = counts.get(fact.level, 0) + 1
            heapq.heappush(order, (-fact.level, -index))
            if not fact.mask:
                heapq.heappush(empty, -index)

        def live(index):
            return pending.get(self.facts[index].cell) == index

        for index in self._parents(cells):
            add(index)
        while pending:
            while empty and not live(-empty[0]):
                heapq.heappop(empty)
            while order and not live(-order[0][1]):
                heapq.heappop(order)
            if empty:
                index = -heapq.heappop(empty)
            else:
                highest, index = -order[0][0], -order[0][1]
                if counts[highest] <= 1:
                    break
                heapq.heappop(order)
            fact = self.facts[index]
            if fact.decision:
                raise ValueError('LCG decision cannot be resolved as an implication.')
            del pending[fact.cell]
            counts[fact.level] -= 1
            if isinstance(fact.parents, _LazyParents):
                expansions += 1
            for parent in fact.parents:
                parent_visits += 1
                add(parent)
            resolutions += 1
        ordered = sorted(pending.values())
        highest = max((self.facts[i].level for i in ordered), default=0)
        uip_index = next((i for i in ordered if self.facts[i].level == highest), None)
        backjump = max((self.facts[i].level for i in ordered if i != uip_index), default=0)
        # Current-path masks for the same cell are nested. Keep their exact
        # conjunction, including mask-valued intermediate facts.
        merged = {}
        for index in ordered:
            fact = self.facts[index]
            merged[fact.cell] = merged.get(fact.cell, self.initial_domains[fact.cell]) & fact.mask
        literals = tuple(sorted(merged.items()))
        uip = None if uip_index is None else (self.facts[uip_index].cell, merged[self.facts[uip_index].cell])
        return LearnedClause(self.binding, literals, uip, highest, backjump, resolutions,
                             expansions, parent_visits, dominated)

    def validate_graph(self):
        for index, fact in enumerate(self.facts):
            if not 0 <= fact.cell < len(self.domains) or fact.mask & ~self.initial_domains[fact.cell]:
                raise ValueError('LCG fact escaped its root.')
            if any(type(p) is not int or not 0 <= p < index for p in fact.parents):
                raise ValueError('LCG graph is not acyclic.')
            inherited = max((self.facts[p].level for p in fact.parents), default=0)
            if (fact.decision and fact.level <= inherited) or (not fact.decision and fact.level != inherited):
                raise ValueError('LCG fact has an inconsistent decision level.')
        return True


class LCGStore:
    """Bounded root-local proved mask clauses; deliberately no disk loader."""
    def __init__(self, fingerprint, target, initial_domains, *, limit=256, literal_limit=MAX_LITERALS):
        self.binding = _binding(fingerprint, target, initial_domains)
        self.fingerprint, self.target, self.initial_domains = self.binding
        if type(limit) is not int or not 1 <= limit <= MAX_CLAUSES:
            raise ValueError('Invalid LCG clause budget.')
        if type(literal_limit) is not int or not 1 <= literal_limit <= MAX_LITERALS:
            raise ValueError('Invalid LCG literal budget.')
        self.limit, self.literal_limit = limit, literal_limit
        self.clauses = []
        self.stats = dict.fromkeys(STAT_KEYS, 0)
        self._prune_snapshot = None
        self._prune_pending = set()
        self._cell_clauses = {}

    def matches(self, fingerprint, target, initial_domains):
        return self.binding == _binding(fingerprint, target, initial_domains)

    def __len__(self):
        return len(self.clauses)

    @staticmethod
    def _subsumes(left, right):
        other = dict(right)
        return all(c in other and not (other[c] & ~mask) for c, mask in left)

    def learn(self, learned):
        if not isinstance(learned, LearnedClause) or learned.binding != self.binding:
            raise ValueError('LCG requires a local proof result bound to the same puzzle and root.')
        merged = {}
        for cell, mask in learned.literals:
            if (type(cell) is not int or not 0 <= cell < len(self.initial_domains) or
                    type(mask) is not int or mask < 0 or mask & ~self.initial_domains[cell]):
                raise ValueError('Invalid LCG learned literal.')
            merged[cell] = merged.get(cell, self.initial_domains[cell]) & mask
        if any(not mask for mask in merged.values()):
            return False  # A conjunction containing an impossible literal.
        clause = tuple(sorted((c, m) for c, m in merged.items() if m != self.initial_domains[c]))
        self.stats['lcg_conflicts'] += 1
        self.stats['lcg_resolutions'] += learned.resolutions
        self.stats['lcg_reason_expansions'] += learned.reason_expansions
        self.stats['lcg_reason_parent_visits'] += learned.parent_visits
        self.stats['lcg_frontier_dominated'] += learned.dominated
        if len(clause) > self.literal_limit:
            self.stats['lcg_budget_skips'] += 1
            return False
        if any(self._subsumes(old, clause) for old in self.clauses):
            self.stats['lcg_subsumed'] += 1
            return False
        retained = [old for old in self.clauses if not self._subsumes(clause, old)]
        self.stats['lcg_subsumed'] += len(self.clauses)-len(retained)
        retained.append(clause)
        self.stats['lcg_learned'] += 1
        while len(retained) > self.limit or sum(map(len, retained)) > self.literal_limit:
            victim = max(range(len(retained)), key=lambda i: (len(retained[i]), -i))
            retained.pop(victim)
            self.stats['lcg_evictions'] += 1
        self.clauses = retained
        self._prune_snapshot = None
        self._prune_pending = set()
        self._cell_clauses = {}
        for index, current in enumerate(retained):
            for cell, _ in current:
                self._cell_clauses.setdefault(cell, set()).add(index)
        return clause in retained

    def asserting_frontier(self, frames, learned, *, require_jump=False):
        """Return an exact pending-frontier replacement for a local UIP proof.

        Frames use the native mask-choice representation. The failed leaf must
        already be consumed. Each frame's unvisited alternatives form a single
        rectangle: its domains with its branch cell restricted to the union of
        the choices at/after its cursor. No completed choice is reintroduced.

        A jump is legal only if *every* deeper pending rectangle entails the
        nogood. Merely reaching the analyzer's backjump level is insufficient
        for DFS: discarded siblings can contain solutions. The retained frame
        must entail every non-UIP literal before asserting the UIP complement.
        If either proof obligation fails, chronological search stays intact.
        This transformation is bounded, allocation-only and has no callbacks;
        callers commit the returned replacement before polling for Stop.
        Native callers may defer a non-jumping unit to normal propagation to
        avoid rebuilding a workspace solely for the next chronological step.
        """
        if not isinstance(learned, LearnedClause) or learned.binding != self.binding:
            raise ValueError('Asserting search requires a root-local LCG proof.')
        if not frames:return None
        if not learned.literals:
            # An empty local proof is unconditional in this effective root.
            self.stats['lcg_assertions']+=1
            self.stats['lcg_asserting_jumps']+=1
            self.stats['lcg_asserting_levels']+=len(frames)
            return []
        if learned.uip is None:return None
        cell, mask = learned.uip
        if (learned.uip not in learned.literals or type(learned.backjump_level) is not int
                or learned.backjump_level < 0 or not 0 <= cell < len(self.initial_domains)
                or mask <= 0 or mask & ~self.initial_domains[cell]):
            raise ValueError('Malformed first-UIP assertion.')
        pending=[]
        for frame in frames:
            domains=list(_masks(frame['domains'], self.initial_domains))
            if frame['cell'] >= 0:
                union=0
                for choice in frame['choices'][frame['next']:]:union |= choice
                domains[frame['cell']] &= union
            pending.append(domains)
        def impossible(domains):
            return not all(domains) or all(not domains[c] & ~bits for c,bits in learned.literals)
        suffix_dead=[True]*(len(pending)+1);suffix_work=[False]*(len(pending)+1)
        for index in range(len(pending)-1,-1,-1):
            suffix_dead[index]=suffix_dead[index+1] and impossible(pending[index])
            suffix_work[index]=suffix_work[index+1] or all(pending[index])
        # A target at the returned level is preferred, but conservative rebasing
        # or an already consumed choice may require a later asserting frame.
        for target in range(min(learned.backjump_level,len(frames)-1),len(frames)):
            domains=pending[target]
            if not suffix_dead[target+1]:continue
            if any(domains[c] & ~bits for c,bits in learned.literals if (c,bits)!=learned.uip):continue
            narrowed=domains[cell] & ~mask
            if narrowed == domains[cell] and all(domains) and target==len(frames)-1:continue
            old=domains[cell]
            # Exhausted rectangles will pop cheaply in either controller. A
            # rewrite must remove real pending boards, not just empty frames.
            if not suffix_work[target+1] and (not all(domains) or old==narrowed):continue
            domains=domains.copy();domains[cell]=narrowed
            replacement=list(frames[:target])
            if all(domains):replacement.append({'domains':domains,'cell':-1,'choices':[],'next':0})
            removed=len(frames)-len(replacement)
            if require_jump and not removed:continue
            self.stats['lcg_assertions']+=1
            self.stats['lcg_asserted_values']+=(old^narrowed).bit_count()
            if removed:
                self.stats['lcg_asserting_jumps']+=1
                self.stats['lcg_asserting_levels']+=removed
            return replacement
        self.stats['lcg_assertion_fallbacks']+=1
        return None

    def prune(self, domains, check, reason=None):
        incoming = _masks(domains, self.initial_domains)
        check()
        if self._prune_snapshot is None:
            candidates = range(len(self.clauses))
        else:
            candidates = set(self._prune_pending)
            for cell, (old, current) in enumerate(zip(self._prune_snapshot, incoming)):
                if old != current:
                    candidates.update(self._cell_clauses.get(cell, ()))
            # An empty nogood forbids every board and has no cell event.
            candidates.update(i for i, clause in enumerate(self.clauses) if not clause)
            candidates = sorted(candidates)
        self.stats['lcg_clause_skips'] += len(self.clauses) - len(candidates)
        # Invalidate before callbacks: cancellation never commits partial work.
        self._prune_snapshot = None
        self._prune_pending = set()
        dirty = set()
        changed = False
        for offset_index, index in enumerate(candidates):
            clause = self.clauses[index]
            if offset_index % 8 == 0:
                check()
            self.stats['lcg_checks'] += 1
            unknown, active = None, True
            for offset, (cell, mask) in enumerate(clause):
                if offset % 32 == 0:
                    check()
                value = domains[cell]
                if not value & mask:
                    active = False
                    break
                if value & ~mask:
                    if unknown is not None:
                        active = False
                        break
                    unknown = (cell, mask)
            if not active:
                continue
            if unknown is None:
                domains[0] = 0
                if reason is not None:
                    reason(0, 0, tuple(c for c, _ in clause))
                self.stats['lcg_hits'] += 1
                return True
            cell, mask = unknown
            domains[cell] &= ~mask
            dirty.update(self._cell_clauses.get(cell, ()))
            if reason is not None:
                reason(cell, domains[cell], tuple(c for c, _ in clause if c != cell))
            self.stats['lcg_prunes'] += 1
            changed = True
        # A later clause can enable an earlier clause. Keep all clauses touched
        # by outputs pending for the next fixpoint call, even when the caller
        # resumes with this exact output. An identical earlier INPUT is also
        # safe: it differs from this saved output and wakes its own events.
        self._prune_snapshot = tuple(domains)
        self._prune_pending = dirty
        return changed
