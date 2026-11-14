"""Reproduce phase, geographic, capacity and Cut&Count comparisons on the encoded puzzle."""
import argparse
import json
from pathlib import Path

from ortools.sat.python import cp_model
from r27_structure_fixtures import independent, planted
from r28_geo_color import dual_cuts, signed_phase_model
from r28_geo_regions import count_partition
from r29_gold import phase_search
from r29_gold_cut_count import cut_witness
from r29_gold_geography import layer_bound, partition_capacity
from r29_gold_phase import EndpointCuts, SignedParity, propagate_rules
from search_problem import SearchProblem
from validator import reverse_mask


def run(tiles, reference, evidence):
    p = SearchProblem(json.loads(tiles.read_text(encoding="utf-8-sig")))
    ref = json.loads(reference.read_text(encoding="utf-8-sig"))
    codes = ref.get("codes",ref.get("residual_reference"))
    bonds = json.loads(evidence.read_text(encoding="utf-8-sig"))["original"]["signed_core"]["bonds"]
    checker = EndpointCuts(p)
    pairs,s2 = checker.clauses(2)
    triples,s3 = checker.clauses(3)
    five = {49:0,62:0,77:0,89:1,126:1}
    twelve = dict(zip((3,13,24,30,31,33,39,61,64,109,124,142),(0,0,0,0,1,0,0,0,0,0,0,0)))
    def older(pins,rules):
        m,x = signed_phase_model(p,pins=pins)
        for rule in rules:
            m.add_bool_or([x[t] if value==0 else ~x[t] for t,value in rule])
        s = cp_model.CpSolver()
        s.parameters.num_search_workers = 1
        s.parameters.max_time_in_seconds = 5
        return s.status_name(s.solve(m))
    phase = dict(pairs=s2,triples=s3,five=five,twelve=twelve,
                 five_old=older(five,()),twelve_with_pairs=older(twelve,pairs),
                 five_blocked=propagate_rules(pairs,five) is None,
                 twelve_blocked=propagate_rules(triples,twelve) is None,
                 symmetric_five=propagate_rules(pairs,{t:1-v for t,v in five.items()}) is None)
    parity = phase_search(p,bonds=bonds,cp=False)
    basis = SignedParity(p)
    mark = len(basis.base.trail)
    for a,b in bonds[:-1]:
        basis.bond(a,b)
    scope = 0
    for a,b in bonds:
        scope |= 1<<(a//3) | 1<<(b//3)
    basis.base.project(scope)
    branch = len(basis.base.trail)
    basis.bond(*bonds[-1])
    basis.base.rollback(branch)
    basis.base.project(scope)
    basis.base.rollback(mark)
    supports = rejected = 0
    for a,(mask,col) in enumerate(checker.sides):
        for b,(other,start) in enumerate(checker.sides[:a]):
            if a//3 == b//3 or reverse_mask(mask) != other:
                continue
            supports += 1
            rejected += not basis.bond(a,b)
            basis.base.rollback(mark)
    parity.update(individual_bonds=supports,individual_pruned=rejected,projection_hits=basis.base.projection_hits)
    layers = {}
    for k in (0,2,6):
        d = [1<<v for v in codes]
        for c in range(k):
            d[c] = sum(7<<(3*(v//3)) for v in codes[:k])
        layers[k] = {name:layer_bound(p,d,detail=name,seconds=10) for name in ("rings","sectors","landmark")}
        layers[k]["gold_vertices"] = layer_bound(p,d,gold=True,detail="vertices",seconds=5)
    ds = [1,1065048,64,512,4096,32768,266240,2097152,16777216,134217728]
    q,_ = planted(1,0,11,True)
    capacity = dict(optimistic=partition_capacity(q,ds,bags=[[]]),separator=partition_capacity(q,ds,bags=[[6]]),
                    independent_solutions=len(independent(q,ds)),residual=dual_cuts(q,ds))
    compare = {}
    for size,free in ((1,0),(1,4),(2,0),(4,0)):
        q,ds = planted(size,free)
        a = list(range(q.n//2))
        b = list(range(q.n//2,q.n))
        cuts = cut_witness(q,ds,a,max_work=150000)
        trie = count_partition(q,[a,b],ds)
        compare[f"{q.n}-{free}"] = dict(cut={k:v for k,v in cuts.items() if k!="witness"},
            witness_valid=bool(cuts.get("witness") and q.validate(cuts["witness"])["valid"]),trie=trie["work"])
    return dict(problem=p.fingerprint,matched=p.validate(codes)["matched_edges"],residual_reference=codes,
                endpoint_rules=phase,parity=parity,layer_bounds=layers,color_capacity=capacity,cut_count=compare)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--tiles",type=Path,default=Path("data/tiles.json"))
    cli.add_argument("--reference",type=Path,default=Path("docs/R29_GOLD_RESULTS.json"))
    cli.add_argument("--r28",type=Path,default=Path("docs/R28_GEOMETRY_RESULTS.json"))
    cli.add_argument("--output",type=Path)
    args = cli.parse_args()
    result = json.dumps(run(args.tiles,args.reference,args.r28),indent=2)
    if args.output:
        args.output.write_text(result+"\n",encoding="utf-8")
    else:
        print(result)
