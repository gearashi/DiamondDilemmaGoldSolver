"""Search ranked supports across a union of centers, sharing score terms and checkpoints."""
import argparse
from collections import Counter, defaultdict
from contextlib import closing, ExitStack
from functools import cached_property, lru_cache
import hashlib
import itertools
import json
import math
from pathlib import Path
import sqlite3
import time

import numpy as np

from r27_incumbent import BoardModel
from r28_exchange import ExchangeBound, combination_batch, domain_binding, read_json
from r28_exchange_rotations import RotationOracle
from r29_exchange_clusters import StarClusters
from r29_exchange_route import existing_balls, is_owned, legal_cell_rotations, owner_copies, physical_hash, rotation_optimum
from r29_move_cores import load_cores
from r30_donor_dd import screened_diagram
from r30_gpu_filters import CudaPageFilters, contact_counts, core_passes
from r30_cycles import FOUR_FAMILIES
from r34_cycle_orders import CycleOrders, FIVE_FAMILIES
from search_problem import SearchProblem


def cycles(order):
    used = set()
    lengths = []
    for i in range(len(order)):
        if i in used:
            continue
        j = i
        length = 0
        while j not in used:
            used.add(j)
            j = order[j]
            length += 1
        if length>1:
            lengths.append(length)
    return tuple(sorted(lengths))


@lru_cache(maxsize=16)
def orders(family):
    ranked = CycleOrders(family)
    # The existing cut masks repeatedly index small four-swap families.
    return tuple(ranked) if family in FOUR_FAMILIES else ranked


def sparse_cycle_type(source, mapping):
    """Cycle lengths between a new endpoint and another center in the same cell coordinates."""
    affected = set(source) | set(mapping)
    inverse = {j:i for i,j in mapping.items()}
    relative = {c:inverse.get(source.get(c,c),source.get(c,c)) for c in affected}
    visited = set()
    lengths = []
    for c in affected:
        if c in visited:
            continue
        length = 0
        while c not in visited:
            visited.add(c)
            c = relative[c]
            length += 1
        if length>1:
            lengths.append(length)
    return tuple(sorted(lengths))


class SharedSupports:
    def __init__(self, model, boards, families, reuse=True, *, bounds=None):
        self.bounds = [ExchangeBound(model,board) for board in boards] if bounds is None else bounds
        self.clusters = [StarClusters(b) for b in self.bounds]
        self.families = tuple(sorted(set(families)))
        self.free = tuple(sorted(model.free))
        self.parity = {sum(f)-len(f) & 1 for f in families}
        self.reuse = reuse
        self.model = model
        self.indices = {cell:i for i,cell in enumerate(self.free)}
        self.copies = []
        for ix,b in enumerate(self.bounds):
            copies = []
            for earlier, donor in enumerate(self.bounds[:ix]):
                size = max(map(sum,self.families))
                for mapping, _, transform in owner_copies(model, [(donor.tiles,size)], b.tiles, size,
                                                         transforms=self.transforms, with_transform=True):
                    if sum(length - 1 for length in cycles(mapping)) % 2 and len(self.parity)==1:
                        continue
                    sparse = {i:j for i,j in enumerate(mapping) if i!=j}
                    copies.append((sparse, earlier, np.argsort(transform)))
            self.copies.append(copies)
        self.fringes = []
        for i,b in enumerate(self.bounds):
            self.fringes.append([])
            for previous in self.bounds[:i]:
                changed = set(np.flatnonzero(previous.tiles!=b.tiles).tolist())
                for _ in range(2):
                    changed.update(int(j) for a in list(changed) for j in b.neighbors[a] if j!=b.n)
                self.fringes[-1].append(frozenset(changed))

    @cached_property
    def transforms(self):
        return legal_cell_rotations(self.model)

    def earlier_location(self, index, cells, order):
        """Return the prior center and ranked support under the same successful rotation."""
        from r32_route_store import colex
        if self.copies[index]:
            mapped = {c:cells[j] for c,j in zip(cells,order)}
            for other, earlier, inverse_cells in self.copies[index]:
                family=sparse_cycle_type(mapped,other)
                if family not in self.families:
                    continue
                changed = [int(inverse_cells[c]) for c in mapped.keys() | other.keys()
                           if mapped.get(c,c) != other.get(c,c)]
                if all(c in self.indices for c in changed):
                    return earlier, colex(changed, self.indices),family
        return None

    def earlier_support(self,index,cells,order):
        located=self.earlier_location(index,cells,order)
        return located[:2] if located is not None else None

    def covered_by_earlier(self, index, cells, order):
        return self.earlier_support(index, cells, order) is not None

    def upper(self,index,cells,values,cache,work):
        b = self.bounds[index]
        if self.reuse:
            for prev,fringe in enumerate(self.fringes[index]):
                if prev in values and fringe.isdisjoint(cells):
                    work['whole_support_reuses'] += 1
                    values[index] = values[prev] - self.bounds[prev].total + b.total
                    return values[index]
        work['support_assignments'] += 1
        values[index] = self.clusters[index].support_upper_twice(cells,cache if self.reuse else None)
        return values[index]


class CountingTerms(dict):
    def __init__(self, work):
        super().__init__()
        self.work = work

    def get(self,key):
        found = super().get(key)
        self.work['local_term_hits' if found is not None else 'local_term_misses'] += 1
        return found


def released_six(binding,physical,families,target):
    """Whether shipped full-family proofs cover every requested physical center."""
    if target<208 or not physical or not families:
        return False
    requested = {tuple(sorted(f)) for f in families}
    if requested-{(2,4),(3,3)}:
        return False
    docs = Path(__file__).parent/'docs'
    try:
        archive = read_json(docs/'R29_EXCHANGE_CENTERS.json')
        results = read_json(docs/'R30_ROUTES_RESULTS.json')
        if results['domain']!=binding or archive['domain']!=binding:
            return False
        recognized = {row['sha256'] for row in archive['centers'] if physical_hash(row['pieces'])==row['sha256']}
        for board in physical:
            center = physical_hash(board)
            if center not in recognized:
                return False
            for family in requested:
                expected = math.comb(len(board),6)*len(orders(family))
                if not any(row['center']==center and tuple(sorted(row['scope']))==family
                           and row['target']==208 and row['hits']==0 and row['complete'] is True
                           and row['total']==expected and row['cursor']==row['end']==len(board)**2
                           for row in results['six']):
                    return False
        return True
    except (KeyError,OSError,TypeError,ValueError):
        return False


def _page_result(db, key, limit, finished, new_pages, work):
    covered = db.execute("SELECT coalesce(sum(hi-lo),0) FROM r30_pages WHERE scope=?", (key,)).fetchone()[0]
    best = None
    for raw, in db.execute("SELECT best FROM r30_pages WHERE scope=? AND best!='null' ORDER BY lo", (key,)):
        item = json.loads(raw)
        if best is None or item['score'] > best['score']:
            best = item
    return dict(scope=key, covered_supports=covered, total_supports=limit, complete=covered==limit,
                range_finished=finished, remaining_supports=limit-covered, best=best, new_pages=new_pages, work=dict(work))


def _covered_prefix(db, key, start, stop):
    """Advance over adjacent committed pages, including a page that straddles start."""
    from r34_rank_leases import covered_end
    return covered_end(db,'r30_pages',key,start,stop)


def run(problem,boards,database,*,families=((2,4),(3,3)),target=208,start=0,stop=None,
        page=64,seconds=0,max_pages=None,filters=('contact','core','connected','star','family','ranges'),share=True,triage=False,backend='cpu',reuse_release=True,progress=None):
    """Negative coverage applies to the union of listed families and centers.

    Pages are colex support ranks. Concurrent workers may submit overlapping ranges:
    workers price leased pages concurrently and atomically publish counts, dependencies and witnesses.
    """
    if backend not in ('cpu','cuda'):
        raise ValueError("Unknown support filter backend.")
    families = tuple(sorted({tuple(sorted(f)) for f in families}))
    if not families or any(f not in FOUR_FAMILIES + FIVE_FAMILIES for f in families):
        raise ValueError("Expected a supported four- or five-swap cycle partition.")
    sizes = {sum(f) for f in families}
    if (len(sizes)!=1 or len(filters)!=len(set(filters)) or set(filters)-{'contact','core','connected','star','family','diagram','ranges','orderprice','shapes'}
            or not 0<=target<=len(problem.edges) or seconds<0 or page<1 or max_pages is not None and max_pages<0):
        raise ValueError("Invalid multi-center search scope or page settings.")
    k, = sizes
    swap_budget = max(k-len(f) for f in families)
    model = BoardModel(problem)
    physical = {}
    for board in boards:
        b = model.legal_board(board)
        physical.setdefault(tuple(map(int,b//3)),b.tolist())
    if not physical:
        raise ValueError("At least one center is required.")
    limit = math.comb(len(model.free),k)
    stop = limit if stop is None else stop
    if not 0<=start<=stop<=limit:
        raise ValueError("Bad support range.")
    binding = domain_binding(model)
    key = hashlib.sha256(json.dumps([1,binding,sorted(physical),families,target]).encode()).hexdigest()
    if reuse_release and released_six(binding,physical,families,target):
        return dict(scope=key,covered_supports=limit,total_supports=limit,complete=True,range_finished=True,
                    evidence='docs/R30_ROUTES_RESULTS.json',remaining_supports=0,best=None,new_pages=0,work={})
    # Page ownership is append-only. A later worker can only fill a remaining hole.
    if Path(database).is_file():
        with closing(sqlite3.connect(database, timeout=90)) as previous:
            if previous.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='r30_pages'").fetchone():
                start = _covered_prefix(previous, key, start, stop)
                if start == stop:
                    return _page_result(previous, key, limit, True, 0, {})
    center_boards = [physical[t] for t in sorted(physical)]
    from r30_bounds import budget_upper, defect_packings
    bounds = [ExchangeBound(model,board) for board in center_boards]
    # Cache center-level bounds separately from the union and target; later jobs can
    # change the center list or cycle family and still use the same valid ceiling.
    with closing(sqlite3.connect(database,timeout=90)) as budgets:
        budgets.execute("CREATE TABLE IF NOT EXISTS r30_budget(cap_key TEXT PRIMARY KEY,cap INTEGER NOT NULL)")
        caps = []
        for bound in bounds:
            ckey = f'{binding}:{swap_budget}:'+physical_hash(bound.tiles)
            saved = budgets.execute("SELECT cap FROM r30_budget WHERE cap_key=?",(ckey,)).fetchone()
            if saved:
                cap = saved[0]
            else:
                cap = budget_upper(bound,defect_packings(bound),swap_budget)
                with budgets:
                    budgets.execute('INSERT OR IGNORE INTO r30_budget VALUES(?,?)',(ckey,cap))
            caps.append(cap)
    if max(caps)<target:
        return dict(scope=key,covered_supports=limit,total_supports=limit,complete=True,range_finished=True,
                    remaining_supports=0,upper_bounds=caps,evidence=f'{swap_budget}-swap defect budget',best=None,new_pages=0,work={})
    graph = SharedSupports(model,center_boards,families,share,bounds=bounds)
    db = sqlite3.connect(database,timeout=90,isolation_level=None)
    t0 = time.monotonic()
    renewal=ExitStack()
    try:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS r30_pages(scope TEXT,lo INTEGER,hi INTEGER,work TEXT,best TEXT,
                PRIMARY KEY(scope,lo));
            CREATE TABLE IF NOT EXISTS optima(key TEXT PRIMARY KEY,result TEXT NOT NULL);
        """)
        from r31_route_coverage import inherited_end, register
        owners = tuple(physical_hash(b.tiles) for b in graph.bounds)
        register(db,key,binding,families,target,owners)
        from r32_route_store import PageProofs, RootCoverage, publish_page, publish_scores
        page_proofs = PageProofs(db, target)
        root_coverage = {(i,f):RootCoverage(db,binding,owner,f,target) for i,owner in enumerate(owners) for f in families}
        deferred = defaultdict(set)
        from r30_cycles import Predecessors, canonical_cycles
        predecessor = {}
        previous_shells = {}
        shape_filters = {}
        undo_checks = {}
        support_indices = {c:i for i,c in enumerate(graph.free)}

        def make_undo(index):
            from r34_exchange_undo import CoveredUndo
            from r32_route_store import colex

            @lru_cache(maxsize=8192)
            def covered(group):
                cells = sorted(c for cycle in group for c in cycle)
                indices = {c:i for i,c in enumerate(cells)}
                moves = {c:d for cycle in group for c,d in zip(cycle,(*cycle[1:],cycle[0]))}
                order = tuple(indices[moves[c]] for c in cells)
                if is_owned(previous_shells[index],cells,order):
                    return True
                family = tuple(sorted(map(len,group)))
                if family not in FOUR_FAMILIES:
                    return False
                rank = colex(cells,support_indices)
                return inherited_end(db,key,binding,(family,),target,(owners[index],),rank,rank+1)>rank

            # A finished lower-target page can give a stronger numeric ceiling
            # than target-1. Only use it after rechecking all owner dependencies.
            thresholds = tuple(row[0] for row in db.execute(
                "SELECT DISTINCT target FROM r31_members WHERE domain=? AND owner=? AND target<? ORDER BY target",
                (binding,owners[index],target)))
            @lru_cache(maxsize=8192)
            def predecessor_upper(group):
                family=tuple(sorted(map(len,group)))
                if thresholds and family in FOUR_FAMILIES:
                    rank=colex([c for cycle in group for c in cycle],support_indices)
                    for threshold in thresholds:
                        if inherited_end(db,key,binding,(family,),threshold,(owners[index],),rank,rank+1)>rank:
                            return threshold-1
                return target-1 if covered(group) else None

            return CoveredUndo(graph.bounds[index],covered,predecessor_upper=predecessor_upper,target=target)

        closed_centers = existing_balls(model,target)
        if closed_centers:
            record = read_json(Path(__file__).parent/'docs/R29_EXCHANGE_RESULTS.json')
            closed_hashes = {r['center'] for r in record['triples'] if r['complete'] is True
                             and r['hits']==0 and r['target']<=target and r['swap_distance']==3
                             and r['moved']==6 and r['objective']=='matched_edges'}
            closed_centers.extend([(c,-1) for c,_ in closed_centers if physical_hash(c) in closed_hashes])
        required = [0]*len(graph.bounds)
        if 'contact' in filters:
            from r30_cycles import certified_contacts
            required = [certified_contacts(b,target,families) for b in graph.bounds]
        from r31_route_geometry import eligible_orders, proved_connected
        geometry = [{f for f in families if 'connected' in filters and proved_connected(b,target,f)}
                    for b in graph.bounds]
        if closed_centers:
            for tiles,radius in list(closed_centers):
                if radius>=0 and released_six(binding,[tiles],[(2,4),(3,3)],target):
                    closed_centers.append((tiles,-2))
        if 'core' in filters:
            from r30_bounds import defect_cores
            cores = []
            for b in graph.bounds:
                legacy = [sum(1<<c for c in core) for core in load_cores(db,model,b.board,target)]
                cores.append(tuple(legacy)+tuple(defect_cores(b,target)))
        else:
            cores = [() for b in graph.bounds]
        oracle = RotationOracle(model)
        orders_by_family = {f:orders(f) for f in families}
        multiplicity = sum(map(len,orders_by_family.values()))
        from r32_route_ranges import PrefixRanges
        intervals = PrefixRanges(graph.free, graph.clusters[0].close, cores, required, geometry,
                                 families, caps, target) if 'ranges' in filters else None
        current, new_pages = start, 0
        previous_report = t0
        work = Counter()
        device_filters = None
        from r34_rank_leases import RankedLeases, LostRankLease
        leases = RankedLeases(db,'support',key,private=database==':memory:',lifetime=renewal)
        busy=False
        while (current<stop or busy) and (max_pages is None or new_pages<max_pages) and (not seconds or time.monotonic()-t0<seconds):
            if current>=stop:
                current=_covered_prefix(db,key,start,stop)
                if current>=stop or max_pages is not None:
                    break
                leases.wait(current,t0+seconds if seconds else float("inf"))
                busy=False
                continue
            db.execute("BEGIN IMMEDIATE")
            prev = db.execute("SELECT hi FROM r30_pages WHERE scope=? AND lo<=? ORDER BY lo DESC LIMIT 1",
                              (key,current)).fetchone()
            if prev and prev[0]>current:
                current = prev[0]
                db.commit()
                continue
            next_lo = db.execute("SELECT min(lo) FROM r30_pages WHERE scope=? AND lo>?",(key,current)).fetchone()[0]
            end = min(stop,current+page,next_lo if next_lo is not None else stop)
            claim,skip=leases.take(current,end)
            if claim is None:
                current=skip
                busy=True
                db.commit()
                continue
            end=claim.hi
            inherited = inherited_end(db,key,binding,families,target,owners,current,end,proofs=page_proofs)
            if inherited>current:
                endpoints = (inherited-current)*multiplicity*len(owners)
                inherited_work = {'first_prior_page':endpoints,'raw_center_endpoints':endpoints}
                db.execute("INSERT INTO r30_pages VALUES(?,?,?,?,?)",
                           (key,current,inherited,json.dumps(inherited_work),'null'))
                if len(owners) > 1:
                    publish_page(db, key, current, inherited, owners, {})
                leases.release(claim)
                db.commit()
                work.update(inherited_work)
                new_pages += 1
                current = inherited
                continue
            db.commit()
            try:
                with leases.computing(claim):
                    counters = Counter()
                    spans = list(intervals.spans(k,current,end,counters)) if intervals else [(current,end)]
                    chunks = [combination_batch(graph.free,k,lo,hi-lo) for lo,hi in spans]
                    cells_batch = np.concatenate(chunks) if chunks else np.empty((0,k),dtype=int)
                    counters['first_rank_interval'] = counters['supports_before_generation'] * len(owners) * multiplicity
                    deferred.clear()
                    from r35_route_dependencies import Dependencies
                    links=Dependencies()
                    page_best = None
                    scores = {}
                    use_cores = 'core' in filters
                    use_contacts = any(required)
                    allowed = pairs = None
                    if len(cells_batch) and (use_cores or use_contacts):
                        if backend == 'cuda':
                            if device_filters is None:
                                device_filters = CudaPageFilters(model.n, cores if use_cores else None,
                                                                 graph.clusters[0].close if use_contacts else None)
                            allowed, pairs = device_filters.evaluate(cells_batch)
                        else:
                            if use_cores:
                                allowed = core_passes(cells_batch, cores, model.n)
                            if use_contacts:
                                pairs = contact_counts(cells_batch, graph.clusters[0].close)
                    # An entire support can be discarded across every center with vector operations.
                    indices = np.arange(len(cells_batch))
                    row_rules = tuple(rule for rule in filters if rule!='ranges')
                    if row_rules and row_rules[0]=='contact' and pairs is not None and min(required)>0:
                        discard = pairs<min(required)
                        counters['first_contact'] += int(discard.sum())*len(required)*multiplicity
                        indices = indices[~discard]
                    for row_index in indices:
                        row = cells_batch[row_index]
                        cells = tuple(map(int,row))
                        local = CountingTerms(counters)
                        geometry_orders = {}
                        upper_by_center = {}
                        survivors = []
                        for ix,b in enumerate(graph.bounds):
                            if caps[ix]<target:
                                counters['first_budget'] += multiplicity
                                continue
                            active_orders = orders_by_family.copy()
                            left = list(families)
                            for rule in filters:
                                if rule=='contact' and required[ix] and pairs[row_index]<required[ix]:
                                    counters['first_contact'] += sum(len(active_orders[f]) for f in left)
                                    left = []
                                if rule=='core' and not allowed[ix,row_index]:
                                    counters['first_core'] += sum(len(active_orders[f]) for f in left)
                                    left = []
                                if rule=='connected':
                                    for f in left:
                                        if f in geometry[ix]:
                                            if f not in geometry_orders:
                                                geometry_orders[f] = tuple(eligible_orders(graph.clusters[0].close,cells,f))
                                            counters['first_connected'] += len(active_orders[f])-len(geometry_orders[f])
                                            active_orders[f] = geometry_orders[f]
                                    left = [f for f in left if active_orders[f]]
                                if left and rule=='orderprice':
                                    retained = []
                                    for f in left:
                                        if f in geometry[ix] and len(active_orders[f]) < len(orders_by_family[f]):
                                            cap = graph.clusters[ix].support_upper_twice(cells,local if share else None,
                                                                                    orders=active_orders[f])
                                            counters['connected_assignment_prices'] += 1
                                            if cap < 2*target:
                                                counters['first_orderprice'] += len(active_orders[f])
                                                continue
                                        retained.append(f)
                                    left = retained
                                if left and rule=='star' and graph.upper(ix,cells,upper_by_center,local,counters)<2*target:
                                    counters['first_support'] += sum(len(active_orders[f]) for f in left)
                                    left = []
                                if left and rule=='family':
                                    retained = []
                                    for f in left:
                                        cap = graph.clusters[ix].support_upper_twice(cells,local if share else None,family=f)
                                        counters['typed_assignments'] += 1
                                        if cap<2*target:
                                            counters['first_typed'] += len(active_orders[f])
                                        else:
                                            retained.append(f)
                                    left = retained
                                if left and rule=='shapes':
                                    from r35_route_shapes import ContactShapes
                                    if ix not in shape_filters:
                                        shape_filters[ix]=ContactShapes(graph.clusters[ix])
                                    shape_filter=shape_filters[ix]
                                    before=shape_filter.work['regional_prices']
                                    for f in left:
                                        if len(f)>1:
                                            remaining=tuple(shape_filter.allowed_orders(cells,active_orders[f],target))
                                            counters['first_shapes']+=len(active_orders[f])-len(remaining)
                                            active_orders[f]=remaining
                                    counters['shape_regional_prices']+=shape_filter.work['regional_prices']-before
                                    left=[f for f in left if active_orders[f]]
                                if left and rule=='diagram':
                                    retained = []
                                    for f in left:
                                        rec = screened_diagram(b,cells,f,target)
                                        counters['dd_transitions'] += rec['transitions'] + rec.get('relaxed_transitions',0)
                                        counters['dd_merged'] += rec['merged']
                                        if rec['upper_twice'] < 2*target:
                                            counters['first_diagram'] += len(active_orders[f])
                                        else:
                                            retained.append(f)
                                    left = retained
                                if not left:
                                    break
                            for f in left:
                                for order in active_orders[f]:
                                    if ix not in previous_shells:
                                        previous_shells[ix] = list(owner_copies(model,closed_centers,b.tiles,k,transforms=graph.transforms) if closed_centers else ())
                                    rc = root_coverage[ix,f]
                                    if rc.scopes and rc.covered(cells,order,graph.clusters[ix].close):
                                        counters['first_root'] += 1
                                        continue
                                    if is_owned(previous_shells[ix],cells,order):
                                        counters['first_owned'] += 1
                                        continue
                                    previous = graph.earlier_location(ix, cells, order)
                                    if previous is not None:
                                        earlier, rank, transformed = previous
                                        from r32_route_store import colex
                                        links.add(owners[ix],f,colex(cells,graph.indices),owners[earlier],rank,transformed)
                                        deferred[owners[ix]].add((owners[earlier], rank))
                                        counters['first_owned'] += 1
                                        continue
                                    group = canonical_cycles(cells,order)
                                    if f in FIVE_FAMILIES:
                                        if ix not in undo_checks:
                                            undo_checks[ix] = make_undo(ix)
                                        reason = undo_checks[ix].rejects(group)
                                    else:
                                        if ix not in predecessor:
                                            predecessor[ix] = Predecessors(b,target)
                                        reason = predecessor[ix].reject(group)
                                    if reason:
                                        counters['first_predecessor'] += 1
                                        continue
                                    upper = graph.clusters[ix].price(cells,order)
                                    if upper < 2*target:
                                        counters['first_endpoint'] += 1
                                    else:
                                        survivors.append((upper,ix,order))
                        if triage:
                            survivors.sort(key=lambda x:-x[0])
                        for _,ix,order in survivors:
                            b = graph.bounds[ix]
                            seed = b.witness_seed(cells,order)
                            if seed is None:
                                counters['illegal'] += 1
                                continue
                            perm = binding+':'+','.join(map(str,seed//3))
                            prior = db.execute('SELECT result FROM optima WHERE key=?',(perm,)).fetchone()
                            if prior:
                                item = json.loads(prior[0])
                                counters['exact_reused'] += 1
                            else:
                                item = rotation_optimum(b,oracle,seed)
                                db.execute('INSERT OR IGNORE INTO optima VALUES(?,?)',(perm,json.dumps(item)))
                                counters['exact_solved'] += 1
                            if item['score']>=target:
                                section = (ix, cycles(order))
                                scores[section] = max(item['score'], scores.get(section,-1))
                                counters['hits'] += 1
                                if page_best is None or item['score']>page_best['score']:
                                    page_best = item
                    endpoints = multiplicity*len(graph.bounds)*(end-current)
                    if sum(counters[s] for s in ('first_budget','first_shapes','first_orderprice','first_rank_interval','first_typed','first_contact','first_core','first_connected','first_support','first_diagram','first_owned',
                                                 'first_root','first_predecessor','first_endpoint','illegal','exact_reused','exact_solved'))!=endpoints:
                        raise RuntimeError("Incomplete page accounting.")
                    counters['raw_center_endpoints'] = endpoints
                    with db:
                        db.execute("BEGIN IMMEDIATE")
                        leases.fence_uncovered(claim,"r30_pages",current,end)
                        db.execute("INSERT INTO r30_pages VALUES(?,?,?,?,?)",
                                   (key,current,end,json.dumps(counters),json.dumps(page_best)))
                        detailed_scores = page_best is not None and (len(owners)>1 or len(families)>1)
                        if len(owners)>1 or detailed_scores:
                            publish_page(db,key,current,end,owners,deferred)
                            links.publish(db,key,current,owners,families)
                        if detailed_scores:
                            publish_scores(db,key,current,owners,families,scores)
                    work.update(counters)
                    current = end
                    new_pages += 1
            except LostRankLease:
                busy=True
                continue
            if progress is not None and time.monotonic()-previous_report>=5:
                covered = db.execute('SELECT sum(hi-lo) FROM r30_pages WHERE scope=?',(key,)).fetchone()[0]
                progress(dict(scope=key,next_support=current,covered_supports=covered,remaining_supports=limit-covered,
                              raw_endpoints_this_run=work['raw_center_endpoints'],exact_solves_this_run=work['exact_solved']))
                previous_report = time.monotonic()
        finished = _covered_prefix(db,key,start,stop)>=stop if busy else current>=stop
        return _page_result(db, key, limit, finished, new_pages, work)
    finally:
        renewal.close()
        db.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tiles',default='data/tiles.json')
    p.add_argument('--database',required=True)
    p.add_argument('--incumbent',action='append',help='Repeat for a union of saved boards; default is the released three.')
    p.add_argument('--family',action='append',choices=[','.join(map(str,f)) for f in FOUR_FAMILIES+FIVE_FAMILIES])
    p.add_argument('--target',type=int,default=208)
    p.add_argument('--start',type=int,default=0)
    p.add_argument('--stop',type=int)
    p.add_argument('--page',type=int,default=64)
    p.add_argument('--seconds',type=float,default=0)
    p.add_argument('--filters',default='contact,core,connected,star,family,ranges')
    p.add_argument('--recheck',action='store_true',help='Ignore the shipped six-cell proof; use a fresh database to reproduce it.')
    p.add_argument('--backend',choices=('cpu','cuda'),default='cpu',help='Support-core filter backend; checkpoints are shared.')
    p.add_argument('--triage',action='store_true')
    args = p.parse_args()
    problem = SearchProblem(read_json(args.tiles))
    if args.incumbent:
        boards = [read_json(name)['codes'] for name in args.incumbent]
    else:
        record = read_json(Path(__file__).parent/'docs/R29_EXCHANGE_CENTERS.json')
        if domain_binding(BoardModel(problem))!=record['domain']:
            p.error('Supply --incumbent for a different puzzle.')
        boards = [[3*t for t in r['pieces']] for r in record['centers']]
    families = [tuple(map(int,f.split(','))) for f in (args.family or ('2,4','3,3'))]
    print(json.dumps(run(problem,boards,args.database,families=families,target=args.target,
                         start=args.start,stop=args.stop,page=args.page,seconds=args.seconds,
                         filters=tuple(filter(None,args.filters.split(','))),triage=args.triage,backend=args.backend,reuse_release=not args.recheck,
                         progress=lambda row: print(json.dumps(row),flush=True))))


if __name__=='__main__':
    main()
