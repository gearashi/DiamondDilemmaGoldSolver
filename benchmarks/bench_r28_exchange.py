"""Reproduce comparisons of exchange certificates and incumbent neighborhoods."""
import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np

from io_utils import atomic_json
from r27_incumbent import BoardModel
from r28_exchange import ExchangeBound, read_json
from r28_exchange_bounds import partial_bound, patch_dominance
from r28_exchange_patches import AssignmentPrices, canonical_paths, cut_trade, joint_patch, shared_blocks, trade_patches
from r28_exchange_rotations import RotationOracle
from search_problem import SearchProblem


def benchmark(problem,codes,seed=27):
    board = np.asarray(codes)
    model = BoardModel(problem)
    pricing = AssignmentPrices(model)
    pricing.update(board)
    rng = np.random.default_rng(seed)
    sample = []
    oracle = RotationOracle(model)
    target = int(model.matches(board).sum())
    ab = ExchangeBound(model,board)
    for idx in range(24):
        cells = rng.choice(model.n,size=2,replace=False)
        neighbor = board.copy()
        neighbor[cells] = neighbor[cells[::-1]]
        score,options = oracle.optimum(neighbor)
        best=next(options)
        report = problem.validate(best)
        if report["matched_edges"]!=score:
            raise RuntimeError("Reference disagrees with rotation solution.")
        pricing.update(neighbor)
        sample.append(score)
    pricing.update(board)
    block = model.independent_set(board,rng)
    independent = []
    for cells in (block[:8],block[:16],block):
        choices = []
        for child in pricing.independent(cells,limit=128):
            rep = problem.validate(child.tolist())
            choices.append((rep["matched_edges"],-rep["line_components"]))
        independent.append(dict(cells=len(cells),ties_examined=len(choices),best=max(choices)))
    defective = [edge for edge,ok in zip(problem.edges,model.matches(board)) if not ok]
    neighbors = problem.neighbors
    anchor = defective[0][0]
    regions = []
    for length in (2,3,4,6,8):
        p = [anchor]
        while len(p)<length:
            candidates = [int(neighbor) for c in p for neighbor in neighbors[c] if neighbor>=0 and neighbor not in p]
            p.append(int(rng.choice(candidates)))
        regions.append(p)
    blocks = []
    for p in regions:
        rows = {c:[3*(int(board[j])//3)+r for j in p for r in range(3)] for c in p}
        now = time.monotonic()
        status,ans = joint_patch(model,board,rows,min_score=target+1,seconds=3)
        joint_status,few = joint_patch(model,board,rows,min_score=target,max_components=51,seconds=3)
        control = list(map(int,block[:len(p)]))
        independent_rows = {c:[3*(int(board[j])//3)+r for j in control for r in range(3)] for c in control}
        control_status,ctrl = joint_patch(model,board,independent_rows,min_score=target+1,seconds=3)
        bounds = [partial_bound(ab,p,assignment=ass,distinct=dis,weights=w)
                  for ass,dis,w in [(False,False,(1,1)),(True,False,(1,1)),(True,True,(1,1)),(True,True,(1,2))]]
        blocks.append(dict(size=len(p),cells=p,patch_threshold=status,independent_threshold=control_status,
            connectivity=joint_status,component_score=None if few is None else problem.validate(few)["line_components"],
            star_bounds=bounds,seconds=time.monotonic()-now))
    # Sweep independent component budgets on the hardest connected block. At
    # 205 and 206, larger disruptions may improve strand connectivity sooner.
    pareto = []
    p = regions[-1]
    rows = {c:[3*(int(board[j])//3)+r for j in p for r in range(3)] for c in p}
    for seam,components in ((target,51),(target-1,51),(target-2,50)):
        now = time.monotonic()
        status,ans = joint_patch(model,board,rows,min_score=seam,max_components=components,
                                 seconds=60 if seam==target else 15)
        point = None
        if ans is not None:
            rep = problem.validate(ans)
            point = [rep["matched_edges"],rep["line_components"]]
        pareto.append(dict(seams=seam,components=components,status=status,actual=point,
                           seconds=time.monotonic()-now))
    disjoint = []
    for (a,sa,b,sb),matched in zip(problem.edges,model.matches(board)):
        if not matched:
            continue
        if not any(c in pair for pair in disjoint for c in (a,b)):
            disjoint.append((a,b))
        if len(disjoint)>=12:
            break
    rigid = []
    for count in (2,3):
        scores = []
        for patches in itertools.islice(itertools.combinations(disjoint,count),64):
            for child in trade_patches(model,board,patches):
                scores.append(int(model.matches(child).sum()))
        rigid.append(dict(count=count,trades=len(scores),best=max(scores)))
    forest = []
    for width in (2,3):
        forest.append(shared_blocks(model,board,disjoint[:width],min_score=target+1,seconds=3)[0])
    cuts = []
    for status,child in cut_trade(model,board,disjoint[:2],list(disjoint[2]),min_score=target+1,seconds=2):
        cuts.append(status)
    # A second layer prices nonoptimal assignments by exact integer seam loss.
    plus_one = [int(model.matches(child).sum()) for child in pricing.independent(block[:12],slack=1,limit=64)]
    patch = disjoint[0]
    placed = board[list(patch)]
    alternatives = [list(t) for t in itertools.product(*[[int(3*(x//3)+r) for r in range(3)] for x in placed])]
    outside = {c:[int(board[c])] for c in range(model.n)}
    dominated = sum(any(patch_dominance(model,outside,patch,cand,other) for other in alternatives)
                    for cand in alternatives)
    orderings = [j for pair in itertools.combinations(disjoint,2) for j in [pair,pair[::-1]]]
    original = problem.validate(board.tolist())
    return dict(start=[target,original["line_components"]],rotated_examples=sample,rotation_work=oracle.work,
        pricing_rows=pricing.rows_done,avoided_price_rows=pricing.rows_reused,independent=independent,
        blocks=blocks,rigid=rigid,patch_dominated=dominated,patch_orientations=len(alternatives),
        unordered_paths=len(list(canonical_paths(orderings))),raw_paths=len(orderings),
        preserved_capped_paths=len(list(canonical_paths(orderings,capped=True))),
        forest=forest,cut_trades=cuts,ranked_scores=plus_one,pareto=pareto)


if __name__=="__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tiles",required=True)
    p.add_argument("--incumbent",required=True)
    p.add_argument("--output",type=Path,required=True)
    args = p.parse_args()
    result=benchmark(SearchProblem(read_json(args.tiles)),read_json(args.incumbent)["codes"])
    atomic_json(args.output,result)
    print(json.dumps(result))
