"""Gold-only necessary connectivity and cut-crossing bounds.

Every tile contains at least one gold segment. A single closed global loop must
visit all cells, so the graph of possibly occupied seams must be connected. If
one possible seam is a bridge of that graph, the loop crosses that cut a
positive even number of times. Its endpoint mask must therefore have at least
two endpoints and even population. The possible graph is an overapproximation:
we intentionally ignore mask agreement and physical-tile competition here.
"""

STAT_KEYS = ('connectivity_checks', 'connectivity_prunes', 'connectivity_failures', 'connectivity_bridges')


class ConnectivityPruner:
    def __init__(self, problem):
        self.n, self.edges = problem.n, tuple(problem.edges)
        self.stats = dict.fromkeys(STAT_KEYS, 0)
        self.nonempty, self.even_positive = [0]*3, [0]*3
        for code, masks in enumerate(problem.masks):
            for side, mask in enumerate(masks):
                count = int(mask).bit_count()
                if count:
                    self.nonempty[side] |= 1 << code
                if count >= 2 and count % 2 == 0:
                    self.even_positive[side] |= 1 << code

    def prune(self, domains, check=lambda: None):
        if any(value == 0 for value in domains):
            return False
        self.stats['connectivity_checks'] += 1
        adjacency = [[] for _ in range(self.n)]
        for index, (a,sa,b,sb) in enumerate(self.edges):
            if index % 32 == 0:
                check()
            if domains[a] & self.nonempty[sa] and domains[b] & self.nonempty[sb]:
                # Edge IDs distinguish parallel seams between the same cells.
                adjacency[a].append((b,index));adjacency[b].append((a,index))
        discovery, low, bridges = [-1]*self.n, [0]*self.n, []
        sequence = 0
        def visit(cell, incoming):
            nonlocal sequence
            check()
            discovery[cell] = low[cell] = sequence; sequence += 1
            for neighbor, edge in adjacency[cell]:
                if edge == incoming:
                    continue
                if discovery[neighbor] < 0:
                    visit(neighbor,edge)
                    low[cell] = min(low[cell],low[neighbor])
                    if low[neighbor] > discovery[cell]:
                        bridges.append(edge)
                else:
                    low[cell] = min(low[cell],discovery[neighbor])
        visit(0,-1)
        if sequence != self.n:
            self.stats['connectivity_failures'] += 1
            self.stats['connectivity_prunes'] += domains[0].bit_count()
            domains[0] = 0
            return True
        self.stats['connectivity_bridges'] += len(bridges)
        changed = False
        for index in bridges:
            check()
            a,sa,b,sb = self.edges[index]
            for cell, side in ((a,sa),(b,sb)):
                old = domains[cell]; new = old & self.even_positive[side]
                if new != old:
                    domains[cell] = new
                    self.stats['connectivity_prunes'] += (old ^ new).bit_count()
                    changed = True
                    if not new:
                        self.stats['connectivity_failures'] += 1
                        return True
        return changed
