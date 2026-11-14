"""Incremental binary Gold search, signed inventory contraction and priced cut bounds."""
from collections import Counter, defaultdict, deque
from itertools import combinations, product

from r27_reason_maxsat import domain_rows
from validator import reverse_mask


class Implications:
    """Reuse the SCC model and the implication closures across Gold-loop branches."""

    def __init__(self, size, clauses, tick):
        from r28_logic_tail import _scc_model
        self.graph = [[] for _ in range(2*size)]
        back = [[] for _ in self.graph]
        for a,b in clauses:
            for x,y in ((a^1,b),(b^1,a)):
                self.graph[x].append(y)
                back[y].append(x)
        self.base = _scc_model(size,self.graph,back,tick)
        self.cache = {}
        self.tick = tick

    def closure(self, literal):
        if literal not in self.cache:
            queue, seen = [literal], 0
            while queue:
                node = queue.pop()
                if not seen & (1 << node):
                    self.tick()
                    seen |= 1 << node
                    queue.extend(self.graph[node])
            self.cache[literal] = seen
        return self.cache[literal]

    def path(self, start, goal):
        """Replayable path for a forced unit (opposite -> selected) or a conflict."""
        seen, todo = {start: None}, deque([start])
        while todo:
            x = todo.popleft()
            if x == goal:
                trail = []
                while x is not None:
                    trail.append(x)
                    x = seen[x]
                return trail[::-1]
            for y in self.graph[x]:
                if y not in seen:
                    todo.append(y)
                    seen[y] = x
        return None

    def assign(self, forced, literal):
        forced |= self.closure(literal)
        even = ((1 << len(self.graph)) - 1) // 3
        return None if (forced & (forced >> 1) & even) else forced

    def model(self, forced):
        return [int(bool(forced & (2 << (2*i)))) if forced & (3 << (2*i)) else v
                for i,v in enumerate(self.base)]


def search_loop_cuts(problem, rows, clauses, target, tick):
    graph = Implications(problem.n, clauses, tick)
    if graph.base is None:
        return None
    def visit(forced):
        tick()
        board = [row[bit] for row,bit in zip(rows,graph.model(forced))]
        report = problem.validate(board)
        if not report['unique_tiles'] or report['matched_edges'] != len(problem.edges):
            raise ArithmeticError("Implication model violated a physical seam or inventory.")
        if target == 'edges' or report['valid']:
            return board
        for c, value in problem.gold_nogood(board):
            literal = 2*c + rows[c].index(value)
            if len(rows[c]) == 1 or forced & (1 << literal):
                continue
            change = graph.assign(forced, literal^1)
            if change is not None:
                witness = visit(change)
                if witness is not None:
                    return witness
            forced = graph.assign(forced, literal)
            if forced is None:
                break
        return None
    return visit(0)


class Parity:
    """Cell i's selected bit equals its representative's bit XOR its relative sign."""
    def __init__(self, size):
        self.p = list(range(size))
        self.sign = [0]*size

    def find(self, i):
        sign = 0
        while i != self.p[i]:
            sign ^= self.sign[i]
            i = self.p[i]
        return i, sign

    def bind(self, i, j, sign):
        a,x = self.find(i)
        b,y = self.find(j)
        if a == b:
            return x^y == sign
        self.p[a], self.sign[a] = b, sign^x^y
        return True


def inventory_map(rows, contract):
    n = len(rows)
    parity = Parity(n+1)
    owners = defaultdict(list)
    compulsory = Counter()
    for c,row in enumerate(rows):
        if len(row) == 1:
            parity.bind(c,n,0)
        if len({x//3 for x in row}) == 1:
            compulsory[row[0]//3] += 1
        else:
            for j,x in enumerate(row):
                owners[x//3].append((c,j))
    for tile in range(n):
        count = compulsory[tile]
        options = owners[tile]
        if count > 1 or not count and not options:
            return None
        if not contract:
            continue
        if count or len(options) == 1:
            if not all(parity.bind(c,n,j^int(bool(count))) for c,j in options):
                return None
        elif len(options) == 2:
            (a,i),(b,j) = options
            if not parity.bind(a,b,1^i^j):
                return None
    pin, offset = parity.find(n)
    ids = {}
    mapping = []
    for c in range(n):
        rep, bit = parity.find(c)
        if rep == pin:
            mapping.append((-1,bit^offset))
        else:
            mapping.append((ids.setdefault(rep,len(ids)),bit))
    return mapping


def cost_terms(problem, rows, mapping, pins):
    terms, unary, const = [], [0]*(1+max(i for i,_ in mapping)), 0
    def value(cell, state):
        i,sign = mapping[cell]
        return rows[cell][(pins.get(i,state if i >= 0 else 0)^sign) % len(rows[cell])]
    for a,sa,b,sb in problem.edges:
        ia,ib = mapping[a][0],mapping[b][0]
        active = list(dict.fromkeys(i for i in (ia,ib) if i >= 0 and i not in pins))
        def defect(x,y):
            return int(int(problem.masks[value(a,x),sa]) != reverse_mask(int(problem.masks[value(b,y),sb])))
        if len(active) < 2:
            low, high = defect(0,0), defect(1,1)
            const += low
            if active:
                unary[active[0]] += high-low
        else:
            terms.append((ia,ib,tuple(defect(i,j) for i,j in product((0,1),repeat=2))))
    return terms, unary, const


def sign_graph(n, terms):
    """Return consistent labels and a cover of sign-inconsistent arcs to condition on."""
    adj = [[] for _ in range(n)]
    for a,b,(x,y,z,w) in terms:
        d = x+w-y-z
        if d:
            adj[a].append((b,int(d>0)))
            adj[b].append((a,int(d>0)))
    signs = {}
    conflicts = set()
    for i in range(n):
        if i in signs:
            continue
        signs[i] = 0
        todo = [i]
        while todo:
            a = todo.pop()
            for b,s in adj[a]:
                if b not in signs:
                    signs[b] = signs[a]^s
                    todo.append(b)
                elif signs[b] != signs[a]^s:
                    conflicts.add(tuple(sorted((a,b))))
    cover = []
    while conflicts:
        count = Counter(i for edge in conflicts for i in edge)
        select = min(count,key=lambda i:(-count[i],i))
        cover.append(select)
        conflicts = {e for e in conflicts if select not in e}
    return signs, cover


def flow_cut(terms, unary, const, signs, active):
    from ortools.graph.python import max_flow
    size = len(unary)
    flow = max_flow.SimpleMaxFlow()
    source,sink = size,size+1
    flow.add_arc_with_capacity(source,sink,0)
    u = list(unary)
    for i, w in enumerate(unary):
        if signs[i]:
            const += w
            u[i] = -w
    for a,b,table in terms:
        aa,ab,ba,bb = (table[2*(i^signs[a])+(j^signs[b])] for i,j in product((0,1),repeat=2))
        delta = ab + ba - aa - bb
        if delta < 0:
            raise ArithmeticError("Unresolved frustration in conditioned graph.")
        const += aa
        u[a] += ba-aa
        u[b] += bb-ba
        if delta:
            flow.add_arc_with_capacity(a,b,delta)
    for i,w in enumerate(u):
        const += min(w,0)
        if w:
            flow.add_arc_with_capacity(i if w<0 else source, sink if w<0 else i, abs(w))
    if flow.solve(source,sink) != flow.OPTIMAL:
        raise ArithmeticError("Failed to certify a cut.")
    after,before = [set() for _ in range(size+2)],[set() for _ in range(size+2)]
    for e in range(flow.num_arcs()):
        a,b = flow.tail(e),flow.head(e)
        if flow.flow(e) < flow.capacity(e):
            after[a].add(b)
            before[b].add(a)
        if flow.flow(e) > 0:
            after[b].add(a)
            before[a].add(b)
    def closure(i,adj):
        visited, stack = {i},[i]
        while stack:
            children = adj[stack.pop()] - visited
            visited.update(children)
            stack.extend(children)
        return visited
    # Every minimum source side is closed under positive residual arcs.
    forced_l,forced_r = closure(source,after),closure(sink,before)
    def ties():
        stack = [(forced_l,forced_r)]
        while stack:
            left,right = stack.pop()
            y = next((i for i in active if i not in left and i not in right),None)
            if y is None:
                yield [signs[i]^int(i not in left) for i in range(size)]
            else:
                up,down = closure(y,after),closure(y,before)
                if not down & left:
                    stack.append((left,right|down))
                if not up & right:
                    stack.append((left|up,right))
    return const+flow.optimal_flow(),ties()


def priced_cut(problem, rows, *, iterations=8, max_ties=128, split_limit=3,
               prices=None, contract=True):
    """Integer-priced inventory relaxation, alternating cycles and conditioned exact cuts.

    A bound stays valid after any cap. Each returned board passes the physical validator.
    Prices are quarter-defect units. They are safe to reuse across neighboring domains.
    """
    from r28_logic_tail import cut_bound
    rows = domain_rows(problem,rows)
    if any(not 1 <= len(row) <= 2 for row in rows):
        raise ValueError("Expected one or two choices per cell.")
    if (type(iterations) is not int or type(max_ties) is not int or type(split_limit) is not int
            or iterations<1 or max_ties<1 or not 0<=split_limit<=12):
        raise ValueError("Invalid binary search limits.")
    seed = list(prices) if prices is not None else [0]*problem.n
    if len(seed)!=problem.n or any(type(x) is not int or abs(x)>100000 for x in seed):
        raise ValueError("Expected one small integer price per tile.")
    mapping = inventory_map(rows, contract)
    if mapping is None:
        return dict(upper=None,score=None,codes=None,gold=False,status="infeasible",prices=seed,work={})
    size = max(i for i,_ in mapping)+1
    stats = Counter(variables=size)
    best_score, best_codes, gold = -1,None,False
    incumbent = cut_bound(problem, rows)
    if incumbent["codes"] is not None:
        best_score,best_codes,gold = incumbent["score"],incumbent["codes"],incumbent["gold"]
    terms,u,c = cost_terms(problem,rows,mapping,{})
    _,cover = sign_graph(size,terms)
    table_upper = incumbent["upper"]
    if len(cover)>split_limit:
        exact = best_codes is not None and best_score == table_upper
        return dict(incumbent,status="optimal" if exact else "relaxed",work=dict(stats),prices=seed)
    best_prices = seed
    branches = []
    for settings in product((0,1),repeat=len(cover)):
        pins = dict(zip(cover,settings))
        terms,u,c = cost_terms(problem,rows,mapping,pins)
        signs,remain = sign_graph(size,terms)
        if remain:
            raise ArithmeticError("Signed cutset failed to restore submodularity.")
        prices = list(best_prices)
        best_lower = None
        stats["conditioned_scopes"] += 1
        active = [i for i in range(size) if i not in pins]
        for step in range(iterations):
            unary = [4*v for v in u]
            const = 4*c-sum(prices)
            for cell,(i,sign) in enumerate(mapping):
                row = rows[cell]
                low = prices[row[(pins.get(i,0)^sign) % len(row)]//3]
                const += low
                if i>=0 and i not in pins:
                    unary[i] += prices[row[(1^sign) % len(row)]//3]-low
            scaled = [(a,b,tuple(4*x for x in t)) for a,b,t in terms]
            lower, ties = flow_cut(scaled,unary,const,signs,active)
            stats["cuts"] += 1
            gradient = None
            attained = False
            exhausted = True
            for index,bits in enumerate(ties):
                if index >= max_ties:
                    exhausted = False
                    break
                stats["tie_models"] += 1
                board = [row[(pins.get(i,bits[i] if i>=0 else 0)^s)%len(row)]
                         for row,(i,s) in zip(rows,mapping)]
                counts = Counter(v//3 for v in board)
                if gradient is None:
                    gradient = [counts[t]-1 for t in range(problem.n)]
                if len(counts)==problem.n:
                    check = problem.validate(board)
                    score = check["matched_edges"]
                    if score > best_score or score == best_score and check['valid']:
                        best_score,best_codes,gold = score,board,bool(check['valid'])
                    attained = True
                    if check['valid'] or score<len(problem.edges):
                        exhausted = False
                        break
            # Every legal board has zero price adjustment. If every optimum cut
            # is illegal, all feasible objective values strictly exceed this cut.
            certified = lower + int(exhausted and not attained)
            if best_lower is None or certified>best_lower:
                best_lower,best_prices = certified,list(prices)
            if attained or (best_score>=0 and len(problem.edges)-(best_lower+3)//4<=best_score):
                break
            prices = [v+d*(2 if step<3 else 1) for v,d in zip(prices,gradient)]
        branches.append(len(problem.edges)-(best_lower+3)//4)
    upper = min(table_upper,max(branches))
    if upper<0 and best_codes is None:
        return dict(upper=None,score=None,codes=None,gold=False,status="infeasible",prices=best_prices,work=dict(stats))
    if best_codes is not None and best_score>upper:
        raise ArithmeticError("A physical board exceeded the proved bound.")
    return dict(upper=upper,score=None if best_score<0 else best_score,codes=best_codes,gold=gold,
                status="optimal" if best_score==upper else "bounded",prices=best_prices,work=dict(stats))
