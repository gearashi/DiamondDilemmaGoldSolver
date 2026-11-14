"""Exact incumbent neighborhoods with an explicit outside boundary and labeled inventory."""
from collections import deque
import itertools

import numpy as np

from constraint_connectivity_round26 import assignment_upper_bound, verify_assignment_bound


class AssignmentPrices:
    """Maintain destination/physical-tile/rotation prices under neighbor edits."""
    def __init__(self, model):
        self.model = model
        self.board = None
        self.prices = np.empty((model.n,model.n,3),dtype=np.int8)
        self.rows_done = 0
        self.rows_reused = 0
        self.dual = None

    def update(self, values):
        board = self.model.legal_board(values)
        nb = self.model.problem.neighbors
        if self.board is None:
            rows = np.arange(self.model.n)
        else:
            changed = np.flatnonzero(board!=self.board)
            rows = np.unique(nb[changed])
            rows = rows[rows>=0]
        self.rows_done += len(rows)
        self.rows_reused += self.model.n-len(rows)
        self.board = board
        for cell in rows:
            table = np.zeros((self.model.n,3),dtype=np.int8)
            for side,other in enumerate(nb[cell]):
                if other >= 0:
                    target = self.model.reverse[self.model.masks[board[other],self.model.problem.sides[cell,side]]]
                    table += (self.model.masks[:,side]==target).reshape(self.model.n,3)
            table[~self.model.legal[cell].reshape(self.model.n,3)] = -10
            self.prices[cell] = table

    def independent(self, cells, *, slack=0, limit=None):
        """Enumerate every assignment within a score gap, including rotation ties."""
        cells = tuple(cells)
        if (not cells or len(set(cells))!=len(cells) or slack<0 or
                set(cells).intersection(self.model.problem.neighbors[list(cells)].ravel())):
            raise ValueError("Expected distinct nonadjacent cells and nonnegative slack.")
        tiles = self.board[list(cells)]//3
        grid = self.prices[list(cells)][:,tiles]
        matrix = [[None if int(v)<0 else int(v) for v in row] for row in grid.max(axis=2)]
        dual = self.dual
        # A stored dual is optimal only if its assignment is still tight/feasible.
        if dual is not None and len(dual["rows"]) == len(cells):
            try:
                verify_assignment_bound(matrix,dual)
                costs = [row[col] for row,col in zip(matrix,dual["assignment"])]
                if None in costs or sum(costs) != dual["bound"]:
                    dual = None
            except ValueError:
                dual = None
        else:
            dual = None
        if dual is None:
            dual = assignment_upper_bound(matrix)
        if dual is None or any(matrix[i][j] is None for i,j in enumerate(dual['assignment'])):
            return
        self.dual = dual
        choices = []
        for i in range(len(cells)):
            row = []
            for j in range(len(tiles)):
                for r,v in enumerate(grid[i,j]):
                    cost = dual["rows"][i]+dual["columns"][j]-int(v)
                    if v >= 0 and cost <= slack:
                        row.append((cost,j,r))
            choices.append(sorted(row))
        sequence = sorted(range(len(cells)),key=lambda i:len(choices[i]))
        candidate = self.board.copy()
        def explore(depth, available, remaining):
            if depth==len(cells):
                if remaining==0:
                    yield candidate.copy()
                return
            i = sequence[depth]
            for cost,j,r in choices[i]:
                if cost > remaining:
                    break
                if (1<<j) & available:
                    candidate[cells[i]] = 3*tiles[j]+r
                    yield from explore(depth+1,available^(1<<j),remaining-cost)
        # Each exact slack layer is generated at most once, monotonically ranked.
        generated = 0
        for gap in range(slack+1):
            for board in explore(0,(1<<len(cells))-1,gap):
                yield board
                generated += 1
                if limit is not None and generated >= limit:
                    return


def shared_blocks(model, values, blocks, *, min_score, max_components=None, seconds=20):
    """Simultaneously optimize every union member using the one shared physical inventory."""
    cells = list(itertools.chain.from_iterable(blocks))
    if len(cells)!=len(set(cells)) or not cells:
        raise ValueError("Shared-inventory blocks must be nonempty and disjoint.")
    board = model.legal_board(values)
    inventory = board[cells]//3
    rows = {c:[int(3*t+r) for t in inventory for r in range(3) if model.legal[c,3*t+r]] for c in cells}
    return joint_patch(model,board,rows,min_score=min_score,max_components=max_components,seconds=seconds)


def cut_trade(model, values, patches, cut, *, min_score, max_components=None, seconds=20):
    """Cycle congruent rigid interiors, then exactly optimize the chosen cut tile inventory."""
    for seed in trade_patches(model, values, patches):
        status,child = shared_blocks(model,seed,[cut],min_score=min_score,
                                     max_components=max_components,seconds=seconds)
        yield status,child


def canonical_paths(paths, *, reads=lambda move: (), capped=False):
    """One trace per commuting order. Capped/intermediate-sensitive search keeps every path.

    A swap writes its two cells; additional cells inspected to make its
    decision are supplied by reads. Preserve relative order of all conflicting
    pairs. Choose the lexicographically least enabled operation otherwise.
    """
    seen = set()
    for path in paths:
        path = tuple(tuple(sorted(move)) for move in path)
        if capped:
            yield path
            continue
        writers = list(map(set,path))
        readers = [set(reads(move)) for move in path]
        predecessors = [set() for _ in path]
        for j in range(len(path)):
            for i in range(j):
                if (writers[i] & (writers[j]|readers[j]) or writers[j] & readers[i]):
                    predecessors[j].add(i)
        pending = set(range(len(path)))
        canonical = []
        while pending:
            i = min((j for j in pending if not predecessors[j] & pending), key=lambda j:(path[j],j))
            pending.remove(i)
            canonical.append(path[i])
        key = tuple(canonical)
        if key not in seen:
            seen.add(key)
            yield key


def rigid_poses(model, values, patch, first_cell):
    """All proper embeddings of an induced connected patch with a chosen anchor."""
    board = model.legal_board(values)
    patch = set(patch)
    if not patch:
        return
    root = min(patch)
    nb, sides = model.problem.neighbors, model.problem.sides
    for rotation in range(3):
        mapping = {root:(first_cell,rotation)}
        image = {first_cell}
        queue = deque([root])
        while queue:
            c = queue.popleft()
            dest,offset = mapping[c]
            for side,neighbor in enumerate(nb[c]):
                if neighbor not in patch:
                    continue
                ds = (side+offset)%3
                other = int(nb[dest,ds])
                if other < 0:
                    break
                row = other,(int(sides[dest,ds])-int(sides[c,side]))%3
                if neighbor in mapping:
                    if mapping[neighbor] != row:
                        break
                elif other in image:
                    break
                else:
                    mapping[neighbor] = row
                    queue.append(neighbor)
                    image.add(other)
            else:
                continue
            break
        else:
            if len(mapping)!=len(patch):
                continue
            if any(any((int(nb[c,s]) in patch) != (int(nb[d,(s+shift)%3]) in image) for s in range(3))
                   for c,(d,shift) in mapping.items()):
                continue
            pose = {d:int(3*(board[c]//3)+(board[c]%3+shift)%3) for c,(d,shift) in mapping.items()}
            if all(model.legal[c,p] for c,p in pose.items()):
                yield pose


def trade_patches(model,board,patches,*,limit=None):
    """Preserve all original patch interiors in a cyclic exchange of chosen patches."""
    regions = [frozenset(p) for p in patches]
    if (len(regions)<2 or any(len(p)!=len(regions[0]) for p in regions)
            or len(set.union(set(),*regions))!=sum(map(len,regions))):
        raise ValueError("Expected disjoint equal-size patches.")
    poses = []
    for source,destination in zip(regions,regions[1:]+regions[:1]):
        choices = []
        for anchor in sorted(destination):
            choices.extend(p for p in rigid_poses(model,board,source,anchor) if p.keys()==destination)
        if not choices:
            return
        poses.append(choices)
    for choices in itertools.islice(itertools.product(*poses),limit):
        child = board.copy()
        for pose in choices:
            for cell,code in pose.items():
                child[cell] = code
        yield child


def joint_patch(model,values,rows,*,min_score,max_components=None,seconds=20):
    """Threshold feasibility with optional selected spanning forest of joined segments.

    rows maps editable cell to allowed orientation codes. The other placements
    are fixed. Exactly one placement per cell and per physical tile is enforced.
    """
    from ortools.sat.python import cp_model
    board = model.legal_board(values)
    choices = [tuple(sorted(set(rows.get(c,[int(board[c])])))) for c in range(model.n)]
    if any(not row or any(not 0<=x<3*model.n or not model.legal[c,x] for x in row)
           for c,row in enumerate(choices)):
        raise ValueError("Invalid joint neighborhood.")
    if max_components is not None and max_components<1:
        raise ValueError("Invalid component limit.")
    cp = cp_model.CpModel()
    picked = [{x:(1 if len(row)==1 else cp.new_bool_var("")) for x in row} for row in choices]
    by_tile = [[] for _ in choices]
    for cell,options in enumerate(picked):
        cp.add(sum(options.values())==1)
        for x,literal in options.items():
            by_tile[x//3].append(literal)
            if len(options)>1:
                cp.add_hint(literal,int(x==board[cell]))
    for uses in by_tile:
        cp.add(sum(uses)==1)
    problem = model.problem
    total = problem.total_segments
    offsets = np.cumsum([0]+list(problem._segment_counts))
    parent = list(range(total))
    def find(x):
        while parent[x]!=x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    links = []
    terms = []
    for a,sa,b,sb in problem.edges:
        for x,la in picked[a].items():
            for y,lb in picked[b].items():
                if x//3==y//3:
                    continue
                matched = model.masks[x,sa]==model.reverse[model.masks[y,sb]]
                local = []
                if max_components is not None:
                    for i,j in zip(problem._segment_ports[x][sa],reversed(problem._segment_ports[y][sb])):
                        if i>=0 and j>=0:
                            local.append((int(offsets[x//3]+i),int(offsets[y//3]+j)))
                if not matched and not local:
                    continue
                if type(la) is int or type(lb) is int:
                    both = lb if type(la) is int else la
                else:
                    both = cp.new_bool_var("")
                    cp.add(both<=la)
                    cp.add(both<=lb)
                    cp.add(both>=la+lb-1)
                if matched:
                    terms.append(both)
                if type(both) is int:
                    for u,v in local:
                        parent[find(u)] = find(v)
                else:
                    links.extend((u,v,both) for u,v in local)
    cp.add(sum(terms)>=min_score)
    if max_components is not None:
        roots = sorted({find(i) for i in range(total)})
        ranks = {r:cp.new_int_var(0,len(roots)-1,"") for r in roots}
        incoming = {r:[] for r in roots}
        for u,v,enabled in links:
            a,b = find(u),find(v)
            if a==b:
                continue
            for x,y in ((a,b),(b,a)):
                selected = cp.new_bool_var("")
                cp.add(selected<=enabled)
                cp.add(ranks[x]<ranks[y]).only_enforce_if(selected)
                incoming[y].append(selected)
        for arcs in incoming.values():
            cp.add(sum(arcs)<=1)
        cp.add(sum(sum(arcs) for arcs in incoming.values())>=len(roots)-max_components)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = seconds
    solver.parameters.num_search_workers = 1
    status = solver.solve(cp)
    if status in (cp_model.FEASIBLE,cp_model.OPTIMAL):
        candidate = [next(x for x,literal in row.items() if solver.value(literal)) for row in picked]
        report = problem.validate(candidate)
        if report["matched_edges"]<min_score or (max_components is not None and report["line_components"]>max_components):
            raise RuntimeError("Independent patch validation disagrees.")
        return solver.status_name(status),candidate
    return solver.status_name(status),None
