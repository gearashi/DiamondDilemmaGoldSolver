"""Search exact physical swap distance with all tile rotations free.

Three disjoint swaps use the native sparse seam graph when available. Other
cycle types and swap distances use the portable search. Select --backend cpu on
compiler-free systems to avoid building the optional native search.
"""
import argparse
import ctypes
import hashlib
import itertools
import json
import math
from pathlib import Path
import sqlite3
import time

import numpy as np
from build_native import _build_one, library_name
from r27_incumbent import BoardModel, rotation_alternatives
from r28_exchange import ExchangeBound, domain_binding, read_json
from r28_exchange_rotations import RotationOracle
from search_problem import SearchProblem

ROOT = Path(__file__).resolve().parent
REVISION = 3


def physical_hash(tiles):
    return hashlib.sha256(str(list(map(int,tiles))).encode()).hexdigest()


def transpositions(order):
    """Minimum swaps realizing a permutation, including fixed points."""
    seen = set()
    result = 0
    for i in range(len(order)):
        if i not in seen:
            j = i
            while j not in seen:
                seen.add(j)
                j = order[j]
                if j != i:
                    result += 1
    return result


def swap_orders(size, distance):
    """Generate fixed-point-free permutations with the required cycle count."""
    def visit(pending, cycles):
        if not pending:
            if not cycles:
                yield tuple(order)
            return
        first, *rest = pending
        for length in range(2, len(pending)-2*(cycles-1)+1):
            if cycles < 1 or (cycles == 1 and length != len(pending)):
                continue
            for others in itertools.permutations(rest, length-1):
                cycle = (first, *others)
                for a,b in zip(cycle, cycle[1:]+cycle[:1]):
                    order[a] = b
                chosen = set(cycle)
                yield from visit([i for i in rest if i not in chosen], cycles-1)
    if not 1 <= size-distance <= size//2:
        return
    order = list(range(size))
    yield from visit(list(range(size)), size-distance)


def rotation_optimum(bound, oracle, seed):
    answer = oracle.optimum(seed)
    if answer is None:
        rows = rotation_alternatives(bound.model,seed,k=1,seconds=120)
        if not rows or not rows[0][1]['exact']:
            raise RuntimeError('Exact rotation search incomplete. Resume this page.')
        board, stats = rows[0]
        score, codes = int(stats['matched_edges']), board.tolist()
    else:
        score, options = answer
        codes = list(map(int,next(options)))
    bound.model.legal_board(codes)
    report = bound.model.problem.validate(codes)
    if report['matched_edges'] != score:
        raise RuntimeError('Independently scored seams disagree.')
    return dict(score=int(score), codes=codes, components=report['line_components'], gold=report['valid'])


def existing_balls(model, target):
    """The released exhaustive neighborhoods have exact domain and center bindings."""
    record = read_json(ROOT/'docs'/'R29_EXCHANGE_CENTERS.json')
    if domain_binding(model)!=record['domain'] or target<record['target']:
        return []
    if any(physical_hash(row['pieces'])!=row['sha256'] for row in record['centers']):
        raise ValueError('Release center fingerprint mismatch.')
    return [(row['pieces'],record['radius']) for row in record['centers']]


def legal_cell_rotations(model):
    """Cell permutations induced by proper rotations that preserve all domains."""
    from constraint_symmetry import proper_rotations
    transforms = []
    problem = model.problem
    if (tuple(map(tuple,problem.neighbors))==problem.board.neighbor_cells and
            tuple(map(tuple,problem.sides))==problem.board.neighbor_sides):
        try:
            for rot in proper_rotations(problem.board):
                if all(set(problem.domains[dest])=={3*(code//3)+(code%3+off)%3
                            for code in problem.domains[c]} for c,(dest,off)
                            in enumerate(zip(rot.cells,rot.side_offsets))):
                    transforms.append(rot.cells)
        except ValueError:
            pass
    if not transforms:
        transforms.append(range(model.n))
    return transforms


def owner_copies(model, centers, current, size):
    """Only retain copies whose radius or completed shell can intersect this shell."""
    transforms = legal_cell_rotations(model)
    seen = set()
    inverse = np.argsort(current)
    for other,radius in centers:
        for perm in transforms:
            oriented = np.empty(model.n,dtype=int)
            oriented[list(perm)] = other
            mapping = inverse[oriented]
            diff = int(np.count_nonzero(mapping!=np.arange(model.n)))
            key = tuple(map(int,mapping)),radius
            limit = radius if radius>=0 else 6
            if diff <= size+limit and key not in seen:
                seen.add(key)
                yield key


def support_owned(owners,cells):
    """Even the farthest possible donor assignment lies inside a closed ball."""
    moved = set(cells)
    return any(radius>=0 and len(moved)+sum(i!=j and i not in moved for i,j in enumerate(other))<=radius
               for other,radius in owners)


def is_owned(owners, moved, order):
    for other,radius in owners:
        changes = {c:c for c,j in enumerate(other) if c!=j}
        changes.update(zip(moved,(moved[j] for j in order)))
        diff = sum(other[c]!=j for c,j in changes.items())
        if radius>=0 and diff<=radius:
            return True
        if radius==-1 and diff==6:
            inverse = {v:k for k,v in enumerate(other)}
            partner = {c:inverse[j] for c,j in changes.items()}
            if all(partner.get(j,j)==c for c,j in partner.items()):
                return True
    return False


def native_library():
    source = ROOT/'r29_exchange_native.c'
    name = 'r29_exchange_native'
    key = hashlib.sha256(source.read_bytes()).hexdigest()[:20]
    folder = ROOT/'.native'/key
    path = folder/library_name(name)
    if not path.is_file():
        _build_one(stem=name,output_dir=folder,manifest_name='route-build.json')
    dll = ctypes.CDLL(str(path))
    dll.route_counter_count.argtypes = []
    dll.route_counter_count.restype = ctypes.c_int
    if dll.route_counter_count()!=6:
        raise RuntimeError('Native route counter ABI differs from Python.')
    i = ctypes.c_int
    u8 = np.ctypeslib.ndpointer(np.uint8,flags='C_CONTIGUOUS')
    i32 = np.ctypeslib.ndpointer(np.int32,flags='C_CONTIGUOUS')
    u64 = np.ctypeslib.ndpointer(np.uint64,flags='C_CONTIGUOUS')
    dll.route_new.argtypes = [i,i,i32,i32,u8,u8,i,i32,i]
    dll.route_new.restype = ctypes.c_void_p
    dll.route_page.argtypes = [ctypes.c_void_p,i,i,u64,i32,i]
    dll.route_free.argtypes = [ctypes.c_void_p]
    dll.route_free.restype = None
    return dll


class NativeTriples:
    def __init__(self,bound,target,owners):
        self.lib = native_library()
        n = bound.n
        if not 0<=target<=len(bound.model.problem.edges):
            raise ValueError('Invalid native target.')
        free = bound.model.free
        self.pairs = np.asarray(list(itertools.combinations(free,2)),dtype=np.int32).reshape(-1,2)
        self.nb = bound.neighbors[:n].astype(np.int32)
        sides = bound.masks[bound.tiles,0,:]
        self.fit = np.ascontiguousarray((sides[:,None,:,None]==bound.model.reverse[sides[None,:,None,:]]).any(axis=(2,3)),dtype=np.uint8)
        cube = bound.support[bound.masks[bound.tiles][...,None],bound.pieces]
        self.masks = np.ascontiguousarray((cube * np.array([1,2,4])[None,:,None,None]).sum(axis=1),dtype=np.uint8)
        self.owners = np.zeros((len(owners),n+14),np.int32)
        for index,(other,radius) in enumerate(owners):
            if sorted(other)!=list(range(n)) or (radius!=-1 and not 0<=radius<=n):
                raise ValueError('Invalid physical ownership region.')
            diff = np.flatnonzero(np.asarray(other)!=np.arange(n))
            if len(diff)>12:
                raise ValueError('Native ownership requires maximum center difference 12.')
            self.owners[index,:n] = other
            self.owners[index,n:n+2] = radius,len(diff)
            self.owners[index,n+2:n+2+len(diff)] = diff
        self.ptr = self.lib.route_new(n,len(self.pairs),self.pairs,self.nb,self.fit,self.masks,
                                      target,self.owners,len(owners))
        if not self.ptr:
            raise MemoryError('Native swap-pair pricing table does not fit.')

    def page(self,start,stop,capacity=2048):
        if (not self.ptr or any(type(v) is not int for v in (start,stop,capacity)) or
                not 0<=start<=stop<=len(self.pairs) or not 1<=capacity<=2**31-1):
            raise ValueError('Invalid or closed native route page.')
        while True:
            counters = np.zeros(6,np.uint64)
            out = np.empty((capacity,6),np.int32)
            count = self.lib.route_page(self.ptr,start,stop,counters,out,capacity)
            if count == -2:
                raise MemoryError('Native exchange workspace allocation failed.')
            if count >= 0:
                return dict(zip(('prefixes','considered','seam_pass','owned','star_pass','prefix_owned'),map(int,counters))),out[:count].copy()
            if stop-start>1:
                mid = (start+stop)//2
                left,a = self.page(start,mid,capacity)
                right,b = self.page(mid,stop,capacity)
                return {k:v+right[k] for k,v in left.items()},np.concatenate((a,b))
            capacity *= 2

    def close(self):
        if self.ptr:
            self.lib.route_free(self.ptr)
            self.ptr = None


def only_certified_contact(clusters, selection, anchors):
    contacts = ((a,b) for a,b in itertools.combinations(selection,2) if clusters.close[a,b])
    first = next(contacts,None)
    return first in anchors and next(contacts,None) is None


def run(problem,codes,target,db,*,distance=3,size=6,backend='native',seconds=0,batch=128,
        reuse_balls=True,max_pages=None,cluster_screen=True):
    from r28_exchange import combination_batch
    if (not 1<=distance<size<=2*distance or size>problem.n or not 0<=target<=len(problem.edges)
            or backend not in {'native','cpu'} or seconds<0 or batch<1):
        raise ValueError('Invalid physical/swap distance, threshold, backend, or budget.')
    bound = ExchangeBound(BoardModel(problem),codes)
    model = bound.model
    binding = domain_binding(model)
    center = bound.tiles.tolist()
    nfree = len(model.free)
    connection = sqlite3.connect(db,timeout=60)
    graph = None
    oracle = RotationOracle(model)
    scope = json.dumps([REVISION,binding,center,distance,size,target,backend,reuse_balls,cluster_screen],separators=(',',':'))
    try:
        connection.execute('CREATE TABLE IF NOT EXISTS route_state(key TEXT PRIMARY KEY, body TEXT)')
        connection.execute('CREATE TABLE IF NOT EXISTS route_closed(binding TEXT,target INTEGER,center TEXT,PRIMARY KEY(binding,target,center))')
        connection.execute('CREATE TABLE IF NOT EXISTS optima(key TEXT PRIMARY KEY, result TEXT NOT NULL)')
        initial = dict(offset=0, prefixes=0, considered=0, seam_pass=0, owned=0, prefix_owned=0, support_owned=0, support_closed=0, separated=0, core_closed=0, star_pass=0, exact=0, reused=0, hits=0, best=None)
        closed = [(json.loads(x[0]),-1) for x in connection.execute(
            'SELECT center FROM route_closed WHERE binding=? AND target<=?',(binding,target))]
        owners = list(owner_copies(model,(existing_balls(model,target) if reuse_balls else [])+closed,
                                   bound.tiles,size))
        from r29_exchange_clusters import StarClusters
        from r29_move_cores import load_cores
        cores = load_cores(connection,model,bound.board,target)
        from r30_bounds import RotationForest, RotationRegions, budget_upper, cell_mask, defect_cores, defect_packings, issued_pair_anchors
        use_defects = cluster_screen and distance==4
        shell_ceiling = paired = forest = anchor_proofs = None
        movement = ()
        clusters = None
        release = read_json(ROOT/'docs'/'R29_EXCHANGE_CENTERS.json')
        separated = (reuse_balls and binding==release['domain'] and target>=release['target'] and any(
            row['pieces']==center and row['separated']['closed'] for row in release['centers']))
        order_count = _order_count(size,size-distance)
        triples = backend=='native' and size==6 and distance==3
        owner_pairs = None
        inner_count = 0
        if triples:
            end = math.comb(nfree,2) if nfree>=6 else 0
        elif separated and cluster_screen and nfree>=size:
            clusters = StarClusters(bound)
            owner_pairs = [(a,b) for a,b in itertools.combinations(model.free,2) if clusters.close[a,b]]
            inner_count = math.comb(nfree-2,size-2)
            end = len(owner_pairs)*inner_count
        else:
            end = math.comb(nfree,size)
        start = time.monotonic()
        pages = 0
        check_six = reuse_balls and size==6 and distance==4 and target>=208
        while True:
            connection.execute('BEGIN IMMEDIATE')
            row = connection.execute('SELECT body FROM route_state WHERE key=?',(scope,)).fetchone()
            state = initial.copy() if row is None else json.loads(row[0])
            pos = state['offset']
            if pos==end or (seconds and time.monotonic()-start>=seconds) or (max_pages is not None and pages>=max_pages):
                connection.commit()
                break
            if check_six and not state['hits']:
                check_six = False
                from r30_multicenter import released_six
                if released_six(binding,[center],((2,4),(3,3)),target):
                    state.update(offset=end,release_closed_cursor=end-pos,
                                 evidence='docs/R30_ROUTES_RESULTS.json')
                    connection.execute('INSERT OR REPLACE INTO route_state VALUES (?,?)',(scope,json.dumps(state)))
                    connection.commit()
                    break
            if use_defects and shell_ceiling is None:
                packings = defect_packings(bound)
                shell_ceiling = budget_upper(bound,packings,distance)
                if shell_ceiling>=target:
                    movement = defect_cores(bound,target,packings)
                    paired = RotationRegions(bound,2)
                    forest = RotationForest(model)
            if shell_ceiling is not None and shell_ceiling<target and not state['hits']:
                # Preserve counts on a partially completed cursor. This proof owns
                # every remaining support and every cycle type at this distance.
                state.update(offset=end,budget_upper=shell_ceiling,budget_closed_cursor=end-pos)
                connection.execute('INSERT OR REPLACE INTO route_state VALUES (?,?)',(scope,json.dumps(state)))
                connection.commit()
                break
            if anchor_proofs is None:
                anchor_proofs = issued_pair_anchors(bound,target,size-2) if cluster_screen and distance>=4 else set()
            step = min(pos+batch,end)
            if triples:
                if graph is None:
                    graph = NativeTriples(bound,target,owners)
                counts,survivors = graph.page(pos,step)
                for k,v in counts.items():
                    state[k] += v
                orders = [((1,0,3,2,5,4),survivors)]
            else:
                if owner_pairs is None:
                    cells = combination_batch(sorted(model.free),size,pos,step-pos)
                else:
                    pair_index,within = divmod(pos,inner_count)
                    step = min(step,(pair_index+1)*inner_count)
                    first = owner_pairs[pair_index]
                    other = [x for x in model.free if x not in first]
                    if size==2:
                        tails = [()]
                    else:
                        tails = combination_batch(other,size-2,within,step-pos)
                    cells = []
                    for tail in tails:
                        moved = tuple(sorted((*first,*tail)))
                        if all((a,b)>=first or not clusters.close[a,b] for a,b in itertools.combinations(moved,2)):
                            cells.append(moved)
                active = []
                for selection in cells:
                    support = cell_mask(selection) if movement or paired is not None else 0
                    if support_owned(owners,selection):
                        state['support_owned'] += order_count
                    elif (any(core.isdisjoint(selection) for core in cores) or
                          any(not support & core for core in movement)):
                        state['core_closed'] += order_count
                    elif cluster_screen:
                        if clusters is None:
                            clusters = StarClusters(bound)
                        if owner_pairs is None and separated and not any(clusters.close[a,b] for a,b in itertools.combinations(selection,2)):
                            state['separated'] += order_count
                        elif anchor_proofs and only_certified_contact(clusters, selection, anchor_proofs):
                            state['family_closed'] = state.get('family_closed',0)+order_count
                        elif clusters.support_upper_twice(selection) - (
                                paired.discount(support) if paired else 0) < 2*target:
                            state['support_closed'] += order_count
                        else:
                            active.append(selection)
                    else:
                        active.append(selection)
                active = np.asarray(active,dtype=int).reshape(-1,size)
                orders = ((order,active) for order in swap_orders(size,distance)) if len(active) else []
            for order,rows in orders:
                if not triples:
                    count = len(rows)
                    rows = np.array([r for r in rows if not is_owned(owners,r,order)],int).reshape(-1,size)
                    state['owned'] += count-len(rows)
                    state['considered'] += len(rows)
                    if len(rows):
                        rows = rows[bound.bounds(rows,order)>=target]
                    state['star_pass'] += len(rows)
                elif len(rows) and np.any(bound.bounds(rows,order)<target):
                    raise RuntimeError('Native and portable star bounds disagree.')
                for moved in rows:
                    seed = bound.witness_seed(moved,order)
                    if seed is None:
                        continue
                    key = binding+':'+','.join(map(str,seed//3))
                    cached = connection.execute('SELECT result FROM optima WHERE key=?',(key,)).fetchone()
                    if cached:
                        item = json.loads(cached[0])
                        state['reused'] += 1
                    else:
                        if forest is not None and forest.upper(seed//3,rounds=1)<target:
                            state['forest_closed'] = state.get('forest_closed',0)+1
                            continue
                        item = rotation_optimum(bound,oracle,seed)
                        connection.execute('INSERT INTO optima VALUES (?,?)',(key,json.dumps(item)))
                        state['exact'] += 1
                    if item['score']>=target:
                        state['hits'] += 1
                        if state['best'] is None or (item['score'],-item['components'])>(state['best']['score'],-state['best']['components']):
                            state['best'] = item
            state['offset'] = step
            connection.execute('INSERT OR REPLACE INTO route_state VALUES (?,?)',(scope,json.dumps(state)))
            if step==end and size==6 and distance==3 and not state['hits']:
                connection.execute('INSERT OR IGNORE INTO route_closed VALUES (?,?,?)',(binding,target,json.dumps(center)))
            connection.commit()
            pages += 1
        return dict(state,complete=state['offset']==end,pages_end=end,center=physical_hash(center),
                    total=math.comb(nfree,size)*order_count,seconds=time.monotonic()-start,
                    objective='matched_edges',swap_distance=distance,moved=size,target=target,cursor='swap' if triples else 'interacting_pair' if owner_pairs is not None else 'support',backend='native' if triples else 'cpu')
    finally:
        if graph is not None:
            graph.close()
        connection.close()


def best_swap_path(problem, initial, final):
    """Maximize the lowest intermediate optimum over all shortest swap paths.

    Every swap in such a path splits one nontrivial cycle of the relative
    permutation. Memoization merges different histories at their physical board.
    Rotations can be reoptimized between swaps; all witness scores are checked.
    """
    from functools import lru_cache
    model = BoardModel(problem)
    bound = ExchangeBound(model,initial)
    target = tuple(model.legal_board(final)//3)
    oracle = RotationOracle(model)
    target_slot = np.argsort(target)
    @lru_cache(None)
    def score(pieces):
        codes = []
        for c,t in enumerate(pieces):
            options = np.flatnonzero(model.legal[c,3*t:3*t+3])
            if not len(options):
                return None
            codes.append(3*int(t)+int(options[0]))
        return rotation_optimum(bound,oracle,codes)
    @lru_cache(None)
    def visit(pieces):
        item = score(pieces)
        if item is None:
            return None
        if pieces==target:
            return item['score'],()
        seen = set()
        best = None
        for i in range(model.n):
            if i in seen:
                continue
            cycle = []
            while i not in seen:
                seen.add(i)
                cycle.append(i)
                i = int(target_slot[pieces[i]])
            for a,b in itertools.combinations(sorted(cycle),2):
                child = list(pieces)
                child[a],child[b] = child[b],child[a]
                reached = visit(tuple(child))
                if reached is not None:
                    floor = min(item['score'],reached[0])
                    path = ((a,b),)+reached[1]
                    if best is None or floor>best[0] or (floor==best[0] and path<best[1]):
                        best = floor,path
        return best
    start = tuple(map(int,bound.tiles))
    result = visit(start)
    if result is None:
        return dict(reachable=False,states=visit.cache_info().misses)
    pieces = list(start)
    boards = [score(start)]
    for a,b in result[1]:
        pieces[a],pieces[b] = pieces[b],pieces[a]
        boards.append(score(tuple(pieces)))
    return dict(reachable=True,floor=result[0],swaps=list(result[1]),boards=boards,states=visit.cache_info().misses)


def _order_count(size,cycles):
    from functools import lru_cache
    @lru_cache(None)
    def count(n,k):
        if n==0:
            return int(k==0)
        if k<1 or 2*k>n:
            return 0
        return sum(math.perm(n-1,length-1)*count(n-length,k-1) for length in range(2,n+1))
    return count(size,cycles)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tiles',default='data/tiles.json')
    p.add_argument('--incumbent',required=True)
    p.add_argument('--database',required=True)
    p.add_argument('--target',type=int,default=208)
    p.add_argument('--swaps',type=int,default=3)
    p.add_argument('--moved',type=int,default=6,help='Physical distance, from swaps+1 to 2*swaps.')
    p.add_argument('--backend',choices=['native','cpu'],default='native')
    p.add_argument('--seconds',type=float,default=0)
    p.add_argument('--batch',type=int,default=128)
    p.add_argument('--no-cluster-screen',action='store_true',help='Portable baseline, with no support-level star screen.')
    p.add_argument('--path',action='store_true',help='Recover the best intermediate score floor of a shortest route to a saved hit.')
    args=p.parse_args()
    problem = SearchProblem(read_json(args.tiles))
    initial = read_json(args.incumbent)['codes']
    result = run(problem,initial,args.target,args.database,distance=args.swaps,size=args.moved,
                 backend=args.backend,seconds=args.seconds,batch=args.batch,cluster_screen=not args.no_cluster_screen)
    if args.path and result['best']:
        result['path'] = best_swap_path(problem,initial,result['best']['codes'])
    print(json.dumps(result))


if __name__=='__main__':
    main()
