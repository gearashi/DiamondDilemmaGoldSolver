"""Reproduce structural qualification on the encoded puzzle and independently checkable controls."""
import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path
from random import Random

from r27_structure_fixtures import nested_loops, planted
from r27_structure_filters import _regions, bits
from r28_geo_color import dual_cuts, phase_probe
from r28_geo_regions import count_partition, paged_trie
from r28_geo_sewing import PosePacking, boundary_lower, ring_census, side_options, sign_filter, supply, switch_family
from search_problem import SearchProblem


def evaluate(tiles, incumbent):
    p = SearchProblem(json.loads(Path(tiles).read_text(encoding="utf-8-sig")))
    g = side_options(p)
    labels = Counter(tuple(map(int,p.masks[v])) for v in range(3*p.n))
    original = dict(signature_classes=len(labels),duplicate_orientations=sum(n-1 for n in labels.values()))
    for euler in (False,True):
        original["phase_euler_" + str(euler)] = phase_probe(p,euler=euler,probe_tiles=range(p.n),seconds=3)
    original["stars"] = ring_census(p,g,max_work=1000000)
    lb,sizes = boundary_lower(p,9,max_shapes=200000)
    original["boundary"] = dict(lower=lb,complete_shape_counts=sizes)
    forced = [(i,next(iter(opts))) for i,opts in enumerate(g) if len(opts) == 1 and min(opts)>i]
    # Connected, scarce-label subsets grown in the real physical option graph.
    neighbors = [Counter() for _ in range(p.n)]
    for a,opts in enumerate(g):
        for b in opts:neighbors[a//3][b//3] += 1/len(opts)
    chosen = set()
    for seed in range(p.n):
        group = {seed}
        for k in range(2,10):
            weights = Counter()
            for t in group:weights.update(neighbors[t])
            weights.subtract({t:1000 for t in group})
            group.add(max(weights,key=lambda x:(weights[x],-x)))
            chosen.add(tuple(sorted(group)))
    minimum = {k:1000 for k in range(2,10)}
    tight = Counter()
    for group in sorted(chosen):
        k = len(group)
        cap = supply(p,group,forced)
        minimum[k] = min(minimum[k],cap)
        if cap <= lb[k]:tight[k] += 1
    original["scarce_subsets"] = dict(tested=len(chosen),min_supply=minimum,saturated=dict(tight))
    poses = PosePacking(p)
    original["forced_pairs"] = len(forced)
    original["forced_poses"] = poses.check(forced)
    original["repeat_poses"] = poses.check(forced)
    original["pose_cache"] = poses.poses.cache_info()._asdict()
    if incumbent:
        raw = json.loads(Path(incumbent).read_text(encoding="utf-8-sig"))
        codes = raw["codes"]
        original["incumbent"] = p.validate(codes)["matched_edges"]
        bonds = []
        for a,s,b,t in p.edges:
            v,w = codes[a],codes[b]
            from validator import reverse_mask
            if int(p.masks[v,s]) == reverse_mask(int(p.masks[w,t])):
                bonds.append((3*(v//3)+(s-v%3)%3,3*(w//3)+(t-w%3)%3))
        from r28_geo_color import phase_bond_core
        original["signed_core"] = phase_bond_core(p,bonds[:100])
        original["signed_core_without_euler"] = phase_bond_core(p,bonds[:100],euler=False)
        for count in (20,100,len(bonds)):
            original["sewn_"+str(count)] = dict(poses=poses.check(bonds[:count]),
                phase=phase_probe(p,bonds=bonds[:count],probe_tiles=range(p.n),seconds=2))

    result = {"original":original}
    controls = {}
    for tag,fixture in (("gold",planted(1,5,23,uniform_ports=True)),
                        ("split",planted(1,5,23,split=True,uniform_ports=True)),
                        ("nested",nested_loops()),("40cells",planted(2,5)),
                        ("160cells",planted(4,7))):
        model,ds = fixture
        current = {"dual":dual_cuts(model,ds),"stars":ring_census(model,side_options(model,ds))}
        if model.n >= 40:
            n = model.board.subdivision
            rings = [[c.index for c in model.board.cells if any(model.board.vertices[v] == ((pole,n),)
                      for v in c.vertices)] for pole in (0,1)]
            parts = [sorted(set(range(model.n))-set(rings[0])-set(rings[1])),rings[0],rings[1]]
        else:parts = [list(range(4)),list(range(4,7)),list(range(7,10))]
        for method in ("plain","trie"):
            r = count_partition(model,parts,ds,mode=method,max_work=1000000)
            current[method] = {k:v for k,v in r.items() if k!="witness"}
        # Check all independently computed labelled counts.
        if current["trie"]["complete"] and current["plain"]["complete"]:
            assert current["trie"]["solutions"] == current["plain"]["solutions"]
        if current["dual"].get("complete") and current["trie"]["complete"]:
            assert current["dual"]["exists"] == bool(current["trie"]["solutions"])
        current["dual"].pop("witness",None)
        controls[tag] = current
    p,_ = planted(1,0,23,uniform_ports=True)
    ds = tuple(sum(1<<(3*t) for t in (range(5) if c<5 else range(5,10))) for c in range(p.n))
    parts = (range(5),range(5,10))
    orbit = {}
    for name in ("plain","trie","response","basis","orbit"):
        out = count_partition(p,parts,ds,mode=name)
        orbit[name] = {k:v for k,v in out.items() if k!="witness"}
    controls["symmetric"] = orbit
    # No equal complete three-side signatures on the original inventory.
    # This direct local switch test evaluates only admissible abstract switch families.
    size = 14
    tour = list(range(1,2*size))+[0]
    switches = [(i*2,i*2+1) for i in range(size)]
    controls["switches"] = switch_family(tour,switches,(0,1,2,3))
    result["controls"] = controls
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiles",type=Path,default=Path("data/tiles.json"))
    parser.add_argument("--incumbent",type=Path)
    parser.add_argument("--output",type=Path)
    args = parser.parse_args()
    value = evaluate(args.tiles,args.incumbent)
    if args.output:
        from io_utils import atomic_json
        atomic_json(args.output,value)
    else:print(json.dumps(value,indent=2))
