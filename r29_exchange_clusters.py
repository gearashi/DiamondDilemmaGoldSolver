"""Portable interaction certificates for physical exchanges at arbitrary distances."""
import hashlib
import json

import numpy as np

from constraint_connectivity_round26 import assignment_upper_bound
from r28_exchange import domain_binding


class StarClusters:
    """Price individual substitutions once. Correct only stars with multiple inputs."""
    def __init__(self, bound):
        self.bound = bound
        n = bound.n
        self.affected = [sorted({c,*map(int,bound.neighbors[c])}-{n}) for c in range(n)]
        self.delta = np.zeros((n,n),int)
        self.local = []
        for c,affected in enumerate(self.affected):
            a = np.asarray(affected)
            central = np.where(a==c,bound.tiles[:,None],bound.tiles[a])
            nb = bound.neighbors[a]
            others = np.where(nb==c,bound.tiles[:,None,None],bound.pieces[nb])
            differences = bound.stars(central,others)-bound.base[a]
            self.delta[c] = differences.sum(axis=1)
            self.local.append(differences)
        b = [set(row) for row in self.affected]
        self.close = np.array([[not x.isdisjoint(y) for y in b] for x in b])
        self.binding = domain_binding(bound.model)+':'+hashlib.sha256(bound.tiles.astype('<i4').tobytes()).hexdigest()

    def price(self,cells,order):
        """Exact star sum for arbitrary exchange size. Corrections are at most fourth order."""
        cells = tuple(map(int,cells))
        donors = [cells[j] for j in order]
        changed = dict(zip(cells,donors))
        counted = {}
        total = self.bound.total + sum(int(self.delta[c,j]) for c,j in zip(cells,donors))
        for c,j in zip(cells,donors):
            for index,v in enumerate(self.affected[c]):
                hits,old = counted.get(v,(0,0))
                counted[v] = hits+1,old+int(self.local[c][j,index])
        mixed = [(v,old) for v,(hits,old) in counted.items() if hits>1]
        if mixed:
            centers = np.asarray([v for v,_ in mixed])
            source = self.bound.pieces.copy()
            for c,j in changed.items():
                source[c] = self.bound.tiles[j]
            total += int((self.bound.stars(source[centers],source[self.bound.neighbors[centers]])-
                          self.bound.base[centers]).sum())-sum(old for _,old in mixed)
        return total

    def support_screen(self,cells,target):
        """Upper-bound every fixed-point-free donor ordering of this support at once.

        A star touched once gets its exact single substitution price. Each
        multiply touched star is either bounded by degree or by maximizing
        over center donor/orientation and independent neighbor donors. The
        resulting prices form a certified physical assignment relaxation.
        """
        cells = tuple(map(int,cells))
        touched = {}
        for c in cells:
            for k,v in enumerate(self.affected[c]):
                touched.setdefault(v,[]).append((c,k))
        const = self.bound.total
        matrix = np.zeros((len(cells),len(cells)),int)
        row = {c:i for i,c in enumerate(cells)}
        for v,hits in touched.items():
            if len(hits)==1:
                c,k=hits[0]
                matrix[row[c]]+=self.local[c][list(cells),k]
                continue
            const -= int(self.bound.base[v])
            centers = [j for j in cells if j!=v] if v in row else [v]
            scores = np.zeros((len(centers),3),int)
            masks = self.bound.masks[self.bound.tiles[centers]]
            for side,nb in enumerate(self.bound.neighbors[v]):
                others = [j for j in cells if j!=nb] if nb in row else [nb]
                scores+=self.bound.support[masks[:,:,side,None],self.bound.pieces[others]].max(axis=-1)
            if v in row:
                for j,best in zip(centers,scores.max(axis=1)):
                    matrix[row[v],row[j]] += best
            else:
                const += int(scores.max())
        shift = int(matrix.min(initial=0))
        values = [[None if i==j else int(v)-shift for j,v in enumerate(line)] for i,line in enumerate(matrix)]
        dual = assignment_upper_bound(values)
        return const + dual['bound']+len(cells)*shift < 2*target

    def separated_certificate(self,target,exact,*,max_leaves=64):
        """Check all legal assignments within dual slack, with a bounded fail-open.

        The certificate covers every support with disjoint one-step stars.
        A missing leaf or solver result keeps the general search enabled.
        """
        n = self.bound.n
        allowed = (~self.close)|np.eye(n,dtype=bool)
        allowed &= self.bound.model.legal.reshape(n,n,3)[:,self.bound.tiles,:].any(2)
        shift = int(self.delta[allowed].min())
        values = [[int(v)-shift if ok else None for v,ok in zip(line,flags)]
                  for line,flags in zip(self.delta,allowed)]
        dual = assignment_upper_bound(values)
        upper = int(self.bound.total+dual['bound']+n*shift)
        slack = upper-2*target
        result = dict(binding=self.binding,target=target,dual_rows=list(dual['rows']),dual_cols=list(dual['columns']),
                      shift=shift,upper_twice=upper,leaves=0,exact=0,witness=None,closed=True)
        if slack<0:
            return result
        deficits = np.array(dual['rows'])[:,None]+np.array(dual['columns'])-(self.delta-shift)
        if np.any(deficits[allowed]<0):
            raise RuntimeError('Infeasible assignment dual.')
        opts = [[(int(col),int(deficits[i,col])) for col in np.flatnonzero(allowed[i] & (deficits[i]<=slack))]
                for i in range(n)]
        chosen = [-1]*n
        def visit(pending,used,moved,gap):
            if not pending:
                if result['leaves']>=max_leaves:
                    result['closed'] = False
                    return
                result['leaves'] += 1
                cells = sorted(moved)
                donors = [chosen[i] for i in cells]
                idx = {c:j for j,c in enumerate(cells)}
                seed = self.bound.witness_seed(cells,[idx[j] for j in donors]) if cells else self.bound.board
                if seed is not None:
                    item = exact(seed)
                    result['exact'] += 1
                    if item['score']>=target:
                        result.update(closed=False,witness=item)
                return
            best = None
            for i in pending:
                available = [(j,cost) for j,cost in opts[i] if cost<=gap and j not in used
                             and (i==j or not any(self.close[i,k] for k in moved))]
                if not available:
                    return
                if best is None or len(available)<len(best[1]):
                    best = i,available
                    if len(available)==1:
                        break
            i,available = best
            for j,cost in available:
                chosen[i] = j
                visit(pending-{i},used|{j},moved|({i} if j!=i else set()),gap-cost)
                if not result['closed']:
                    return
        visit(set(range(n)),set(),set(),slack)
        return result


def interacting_supports(pool,size,close):
    """Generate only size-k supports containing an interacting pair.

    Colexicographic ranks are not used here. Each support's first intersecting
    pair in sorted order owns it, so two intersecting pairs cannot duplicate work.
    """
    import itertools
    pool=tuple(sorted(pool))
    if size<2:
        return
    for i,j in itertools.combinations(pool,2):
        if not close[i,j]:
            continue
        others = [c for c in pool if c!=i and c!=j and
                  not (c<i and (close[c,i] or close[c,j])) and not (i<c<j and close[i,c])]
        for extra in itertools.combinations(others,size-2):
            if any(close[a,b] and (a,b)<(i,j) for a,b in itertools.combinations(extra,2)):
                continue
            yield tuple(sorted((i,j,*extra)))
