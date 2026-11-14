"""Reproduce Round 34 physical matching, path, pricing and channel comparisons."""
import argparse
from collections import Counter
import json
from random import Random

from r27_structure_filters import Work, _perfect_supports, domain_bits
from r28_exchange import read_json
from r30_gold import GoldMatching
from r31_geographic import channel_geographic, physical_bonds
from r31_gold_pricing import _odd_mate_sets
from r32_gold_bonds import gold_matching_supports, inventory
from search_problem import SearchProblem


def pin_residuals(problem, rules, board):
    reference = sorted(set(physical_bonds(problem, board)) & rules.possible)
    report = []
    for count in (4,12,24,48,96,144):
        counts = Counter()
        for seed in range(10):
            pins = Random(seed).sample(reference, count)
            fixed = dict(pins)
            fixed.update({b:a for a,b in pins})
            rows = [set() for _ in range(3*problem.n)]
            for a,b in rules.possible:
                if fixed.get(a,b)==b and fixed.get(b,a)==a:
                    rows[a].add(b)
                    rows[b].add(a)
            baseline = _perfect_supports(rows,Work(2000000))
            if baseline is None:
                counts['matching_infeasible'] += 1
                continue
            trial = gold_matching_supports(problem,rows,Work(2000000))
            if trial is None:
                counts['extra_infeasible'] += 1
            else:
                counts['extra_mates_removed'] += sum(len(a)-len(b) for a,b in zip(baseline,trial))//2
        report.append(dict(pins=count,**counts))
    return report


def blank_side_pricing(problem, rules):
    from ortools.linear_solver import pywraplp
    sides = [3*t+s for t,row in enumerate(problem.masks[::3]) for s,v in enumerate(row) if not int(v)]
    left = set(sides[:len(sides)//2])
    def crossing(pair, part):
        return (pair[0] in part)!=(pair[1] in part)
    rng = Random(34)
    def cycle(vertices):
        for _ in range(10000):
            rng.shuffle(vertices)
            edges={tuple(sorted((v,vertices[i-1]))) for i,v in enumerate(vertices)}
            if edges<=rules.possible:
                return edges
        raise ValueError('Expected the released blank-side class')
    links=cycle(sorted(left))|cycle(sorted(set(sides)-left))
    links.add(next(pair for pair in sorted(rules.possible)
                   if all(v in sides for v in pair) and crossing(pair,left)))
    rows=[set() for _ in sides]
    for a,b in links:
        x,y=sides.index(a),sides.index(b)
        rows[x].add(y)
        rows[y].add(x)
    work=Work(50000)
    reduced=_perfect_supports(rows,work)
    lp=pywraplp.Solver.CreateSolver('GLOP')
    x={pair:lp.NumVar(0,1,'') for pair in sorted(links)}
    for u in sides:
        lp.Add(sum(v for pair,v in x.items() if u in pair)==1)
    lp.Minimize(sum(v for pair,v in x.items() if crossing(pair,left)))
    assert lp.Solve()==lp.OPTIMAL
    before=lp.Objective().Value()
    weights={pair:v.solution_value() for pair,v in x.items()}
    cuts=[]
    for part in _odd_mate_sets(sides,weights,100000):
        if sum(v for pair,v in weights.items() if crossing(pair,part))<1-1e-6:
            lp.Add(sum(v for pair,v in x.items() if crossing(pair,part))>=1)
            cuts.append(len(part))
    assert lp.Solve()==lp.OPTIMAL
    return dict(before=before,after=lp.Objective().Value(),cuts=cuts,candidates=len(links),
                exact_mates=sum(map(len,reduced))//2,matching_states=work.counters['partner_matching_states'])


def skeleton_hall(problem):
    from r32_fresh_skeleton import parity_seed
    from r32_fresh_geometry import skeletons
    board=parity_seed(problem,34)['codes']
    rng=Random(34)
    report=[]
    for free in (16,32,48):
        pool=[3*(v//3)+r for v in board[:free] for r in range(3)]
        domains=[]
        for c,v in enumerate(board):
            row=1<<v
            if c<free:
                row |= sum(1<<j for j in rng.sample([i for i in pool if i!=v],4))
            domains.append(row)
        def run(early):
            return skeletons(problem,domains,max_nodes=500000,limit=50000,early_hall=early)
        old,trial=run(False),run(True)
        assert old['complete'] and trial['complete'] and old['patterns']==trial['patterns']
        report.append(dict(free=free,before=old['work']['states'],after=trial['work']['states'],
                           cuts=trial['work']['early_hall_rejections'],patterns=len(trial['patterns'])))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiles',default='data/tiles.json')
    args=parser.parse_args()
    problem=SearchProblem(read_json(args.tiles))
    rules=inventory(problem)
    ds=domain_bits(problem)
    check=GoldMatching(problem,ds,{},[])
    channel=channel_geographic(check)
    if check.m.validate():
        raise AssertionError(check.m.validate())
    board=read_json('docs/R29_INCUMBENT.json')['codes']
    print(json.dumps(dict(root=dict(possible=len(rules.possible),required=len(rules.required),forbidden=len(rules.forbidden)),
                          skeleton_hall=skeleton_hall(problem),residuals=pin_residuals(problem,rules,board),blank_class=blank_side_pricing(problem,rules),
                          geographic=dict(channel=channel,variables=len(check.m.proto.variables),
                                          constraints=len(check.m.proto.constraints),
                                          pose_pair_products=sum(ds[a].bit_count()*ds[b].bit_count()
                                                                 for a,s,b,t in problem.edges))),indent=2))


if __name__=='__main__':
    main()
