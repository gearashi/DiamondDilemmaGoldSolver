"""Two optional swap samplers. The Metropolis clock counts rejected proposals."""
from functools import lru_cache
import math

import numpy as np

from r28_swap_walk import PairScores


@lru_cache(maxsize=8192)
def discrepancy(left, right):
    """Minimum endpoint shift cost, charging six units for a missing endpoint."""
    a = [i for i in range(11) if left >> i & 1]
    b = [i for i in range(11) if right >> i & 1]
    row = [6*i for i in range(len(b)+1)]
    for i, x in enumerate(a, 1):
        new = [6*i]
        for j, y in enumerate(b, 1):
            new.append(min(row[j]+6, new[-1]+6, row[j-1]+abs(x-y)))
        row = new
    return row[-1]


def candidate(evaluator, board, index):
    p, r, s = np.unravel_index(index, (len(evaluator.left),3,3))
    a, b = evaluator.left[p], evaluator.right[p]
    child = board.copy()
    child[a], child[b] = board[b]//3*3+r, board[a]//3*3+s
    return child


def mismatch_cost(model, board):
    return sum(discrepancy(int(a),int(b)) for a,b in zip(model.masks[board[model.a], model.sa],
                    model.reverse[model.masks[board[model.b], model.sb]]) if a != b)


def plateau_step(evaluator, board, rng, *, samples=16, displacements=True):
    scores = evaluator.all(board).ravel()
    best = scores.max()
    if best < 0:
        return board.copy()
    ties = np.flatnonzero(scores == best)
    chosen = rng.choice(ties, size=min(samples, len(ties)), replace=False)
    options = (candidate(evaluator,board,i) for i in chosen)
    if displacements:
        return min(options,key=lambda b:mismatch_cost(evaluator.model,b))
    return next(options)


def jump(evaluator, board, rng, temperature, remaining):
    """Exact constant-temperature uniform oriented-swap Metropolis jump."""
    if not math.isfinite(temperature) or temperature < 0 or type(remaining) is not int or remaining < 1:
        raise ValueError("Invalid temperature or proposal clock")
    scores = evaluator.all(board).ravel()
    score = int(evaluator.model.matches(board).sum())
    if not scores.size:
        return board.copy(), remaining, False
    if temperature == 0:
        weights = (scores >= score).astype(float)
    else:
        weights = np.exp(np.minimum(0, scores.astype(float)-score)/temperature)
        weights[scores < 0] = 0.
    total = weights.sum()
    if total == 0:
        return board.copy(), remaining, False
    probability = min(1., total/len(scores))
    # Draw the geometric holding time in log space. A truncated clock avoids
    # overflow for very small probabilities and includes the accepted attempt.
    u = rng.random()
    log_survival = math.log1p(-probability) if probability < 1 else -math.inf
    if probability < 1 and math.log1p(-u) <= remaining*log_survival:
        return board.copy(), remaining, False
    wait = 1 if probability == 1 else 1+int(math.log1p(-u)/log_survival)
    index = rng.choice(len(scores), p=weights/total)
    return candidate(evaluator,board,index), wait, True


class JumpChain:
    """Cache the complete integer move spectrum between accepted swaps."""
    def __init__(self, model, board, temperature):
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("Invalid temperature")
        self.pairs = PairScores(model)
        self.model = model
        self.board = model.legal_board(board)
        self.temperature = temperature
        self.score = int(model.matches(self.board).sum())
        grid = self.pairs.all(self.board).astype(np.int16)
        self.deltas = np.where(grid >= 0, grid-self.score, -1000)
        self.evaluations = grid.size
        self.steps = self.accepted = 0
        self.best = self.board.copy()
        self.best_score = self.score

    def advance(self, rng, proposals):
        if type(proposals) is not int or proposals < 0:
            raise ValueError("Invalid proposal bound")
        remaining = proposals
        while remaining and self.deltas.size:
            flat = self.deltas.ravel()
            if self.temperature == 0:
                eligible = np.flatnonzero(flat >= 0)
                chance = len(eligible)/len(flat)
            else:
                weights = np.exp(np.minimum(flat.astype(float),0.)/self.temperature)
                weights[flat == -1000] = 0.
                total = weights.sum()
                chance = min(1.,total/len(flat))
            if chance <= 0:
                break
            log_u = math.log1p(-rng.random())
            log_q = math.log1p(-chance) if chance < 1. else -math.inf
            if log_u <= remaining*log_q:
                break
            wait = 1 if chance == 1. else 1+int(log_u/log_q)
            remaining -= wait
            index = rng.choice(eligible) if self.temperature == 0 else rng.choice(len(flat),p=weights/total)
            pair = index//9
            self.score += int(flat[index])
            self.board = candidate(self.pairs,self.board,index)
            self.accepted += 1
            if self.score > self.best_score:
                self.best, self.best_score = self.board.copy(), self.score
            a,b=self.pairs.left[pair],self.pairs.right[pair]
            affected = np.zeros(self.model.n,dtype=bool)
            neighbors = self.model.problem.neighbors[[a,b]].ravel()
            affected[neighbors[neighbors>=0]] = True
            affected[[a,b]] = True
            rows = np.flatnonzero(affected[self.pairs.left]|affected[self.pairs.right])
            refreshed = self.pairs.all(self.board,rows)
            self.deltas[rows] = np.where(refreshed>=0,refreshed-self.score,-1000)
            self.evaluations += refreshed.size
            if self.score != int(self.model.matches(self.board).sum()):
                raise AssertionError('Accepted score differs from direct seam evaluation')
        self.steps += proposals
        return self.board.copy()


def direct_stepper(evaluator, board, rng, temperature, remaining):
    # Reference uses exactly the same labeled pair/rotation proposal law.
    score = int(evaluator.model.matches(board).sum())
    for clock in range(1,remaining+1):
        child = candidate(evaluator,board,rng.integers(len(evaluator.left)*9))
        if not evaluator.model.legal[np.arange(len(child)),child].all():
            continue
        delta = int(evaluator.model.matches(child).sum()) - score
        if delta >= 0 or temperature and rng.random() < math.exp(delta/temperature):
            return child, clock, True
    return board.copy(), remaining, False

def main():
    import argparse,json
    from pathlib import Path
    from io_utils import atomic_json
    from r27_incumbent import BoardModel
    from search_problem import SearchProblem
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('tiles',type=Path)
    p.add_argument('board',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--proposals',type=int,default=1000000)
    p.add_argument('--temperature',type=float,default=0.05)
    p.add_argument('--seed',type=int,default=0)
    a=p.parse_args()
    model=BoardModel(SearchProblem(json.loads(a.tiles.read_text(encoding='utf-8-sig'))))
    chain=JumpChain(model,json.loads(a.board.read_text(encoding='utf-8-sig'))['codes'],a.temperature)
    chain.advance(np.random.default_rng(a.seed),a.proposals)
    codes=chain.best.tolist()
    report=model.problem.validate(codes)
    atomic_json(a.output,dict(codes=codes,solved=report['valid'],validation=report))
    print(json.dumps(dict(best=chain.best_score,score=chain.score,accepted=chain.accepted,clock=chain.steps,move_evaluations=chain.evaluations)))


if __name__ == '__main__':
    main()
